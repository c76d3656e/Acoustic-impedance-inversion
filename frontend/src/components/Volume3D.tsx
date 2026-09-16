import { useEffect, useMemo, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useShallow } from "zustand/react/shallow";
import * as THREE from "three";
import { useStore } from "../store";
import { buildLUT } from "../viz/colormaps";
import { extractSlice, sliceToImageData } from "../viz/slice";
import type { FieldVolume, Manifest, SliceAxis } from "../types";

const S = 1 / 100; // meters -> world units

/* ---------- volume raymarch shaders (WebGL2 / GLSL3) ---------- */
const VERT = /* glsl */ `
out vec3 vOrigin;
out vec3 vDir;
void main() {
  vec3 p = position + 0.5;                       // unit cube [0,1]
  vec3 camLocal = (inverse(modelMatrix) * vec4(cameraPosition, 1.0)).xyz + 0.5;
  vOrigin = camLocal;
  vDir = p - camLocal;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const FRAG = /* glsl */ `
precision highp float;
precision highp sampler3D;
uniform sampler3D uData;
uniform sampler2D uLUT;
uniform float uOpacity;
uniform float uSteps;
uniform int uClipOn;
uniform int uClipAxis;
uniform float uClipPos;
uniform int uClipSide;
in vec3 vOrigin;
in vec3 vDir;
out vec4 outColor;

vec2 hitBox(vec3 o, vec3 d) {
  vec3 invd = 1.0 / d;
  vec3 t0 = (vec3(0.0) - o) * invd;
  vec3 t1 = (vec3(1.0) - o) * invd;
  vec3 tmin = min(t0, t1), tmax = max(t0, t1);
  return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

void main() {
  vec3 rd = normalize(vDir);
  vec2 b = hitBox(vOrigin, rd);
  b.x = max(b.x, 0.0);
  if (b.x > b.y) discard;
  float dt = (b.y - b.x) / uSteps;
  vec3 p = vOrigin + rd * b.x;
  vec4 acc = vec4(0.0);
  for (int i = 0; i < 400; i++) {
    if (float(i) >= uSteps) break;
    bool clipped = false;
    if (uClipOn == 1) {
      float c = uClipAxis == 0 ? p.x : (uClipAxis == 1 ? p.y : p.z);
      clipped = (uClipSide == 1) ? (c > uClipPos) : (c < uClipPos);
    }
    if (!clipped) {
      vec3 sc = vec3(p.x, p.y, 1.0 - p.z);
      float v = texture(uData, sc).r;
      vec3 col = texture(uLUT, vec2(clamp(v, 0.001, 0.999), 0.5)).rgb;
      float a = clamp(uOpacity * dt * 26.0 * (0.4 + v), 0.0, 1.0);
      acc.rgb += (1.0 - acc.a) * col * a;
      acc.a += (1.0 - acc.a) * a;
      if (acc.a > 0.97) break;
    }
    p += rd * dt;
  }
  if (acc.a < 0.01) discard;
  outColor = vec4(acc.rgb, acc.a);
}
`;

/* ---------- helpers for the opaque cut-face quad ---------- */
function corners(m: Manifest, axis: SliceAxis, index: number) {
  const { x, y, z } = m.axes;
  const X = (v: number) => v * S, Y = (v: number) => v * S, Z = (v: number) => v * S;
  const xN = x.length - 1, yN = y.length - 1, zN = z.length - 1;
  if (axis === "z") {
    const zc = Z(z[index]);
    return [
      [X(x[0]), Y(y[0]), zc], [X(x[xN]), Y(y[0]), zc],
      [X(x[xN]), Y(y[yN]), zc], [X(x[0]), Y(y[yN]), zc],
    ];
  }
  if (axis === "x") {
    const xc = X(x[index]);
    return [
      [xc, Y(y[0]), Z(z[zN])], [xc, Y(y[yN]), Z(z[zN])],
      [xc, Y(y[yN]), Z(z[0])], [xc, Y(y[0]), Z(z[0])],
    ];
  }
  const yc = Y(y[index]);
  return [
    [X(x[0]), yc, Z(z[zN])], [X(x[xN]), yc, Z(z[zN])],
    [X(x[xN]), yc, Z(z[0])], [X(x[0]), yc, Z(z[0])],
  ];
}

function writeQuad(geom: THREE.BufferGeometry, c: number[][]) {
  const [bl, br, tr, tl] = c;
  const p = geom.getAttribute("position") as THREE.BufferAttribute;
  (p.array as Float32Array).set([
    ...bl, ...br, ...tr, ...bl, ...tr, ...tl,
  ]);
  p.needsUpdate = true;
  geom.computeBoundingSphere();
}

function drawSlice(
  canvas: HTMLCanvasElement, field: FieldVolume, m: Manifest,
  axis: SliceAxis, index: number, lut: Uint8Array,
) {
  const slice = extractSlice(field, m, axis, index);
  const img = sliceToImageData(slice, lut, field.meta.min, field.meta.max);
  canvas.width = slice.w;
  canvas.height = slice.h;
  canvas.getContext("2d")!.putImageData(img, 0, 0);
}

function buildData3D(field: FieldVolume, m: Manifest): THREE.Data3DTexture {
  const { nx, ny, nz } = m.grid;
  const { min, max } = field.meta;
  const span = max - min || 1;
  const arr = new Uint8Array(nx * ny * nz);
  const d = field.data;
  for (let ix = 0; ix < nx; ix++)
    for (let iy = 0; iy < ny; iy++)
      for (let iz = 0; iz < nz; iz++) {
        const v = d[(ix * ny + iy) * nz + iz];
        arr[ix + iy * nx + iz * nx * ny] = Math.max(0, Math.min(255,
          Math.round(((v - min) / span) * 255)));
      }
  const tex = new THREE.Data3DTexture(arr, nx, ny, nz);
  tex.format = THREE.RedFormat;
  tex.type = THREE.UnsignedByteType;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  return tex;
}

function buildLUTTexture(lut: Uint8Array): THREE.DataTexture {
  const rgba = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    rgba[i * 4] = lut[i * 3];
    rgba[i * 4 + 1] = lut[i * 3 + 1];
    rgba[i * 4 + 2] = lut[i * 3 + 2];
    rgba[i * 4 + 3] = 255;
  }
  const tex = new THREE.DataTexture(rgba, 256, 1, THREE.RGBAFormat);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}

function Scene() {
  const manifest = useStore((s) => s.manifest);
  const field = useStore((s) => s.currentField());
  const wells = useStore((s) => s.wells);
  const colormap = useStore((s) => s.colormap);
  const reverse = useStore((s) => s.reverse);
  const showBoreholes = useStore((s) => s.showBoreholes);
  const volumeOpacity = useStore((s) => s.volumeOpacity);
  const sectionOn = useStore((s) => s.sectionOn);
  const sectionAxis = useStore((s) => s.sectionAxis);
  const sectionIndex = useStore((s) => s.sectionIndex);
  const sectionReverse = useStore((s) => s.sectionReverse);
  const grid = useStore(useShallow((s) => s.manifest?.grid));

  const lut = useMemo(() => buildLUT(colormap, reverse), [colormap, reverse]);

  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        glslVersion: THREE.GLSL3,
        uniforms: {
          uData: { value: null },
          uLUT: { value: null },
          uOpacity: { value: 0.45 },
          uSteps: { value: 160 },
          uClipOn: { value: 0 },
          uClipAxis: { value: 0 },
          uClipPos: { value: 0.5 },
          uClipSide: { value: 1 },
        },
        vertexShader: VERT,
        fragmentShader: FRAG,
        transparent: true,
        side: THREE.BackSide,
        depthWrite: false,
      }),
    [],
  );

  const sectionRef = useRef<{ mesh: THREE.Mesh; geom: THREE.BufferGeometry; tex: THREE.CanvasTexture; canvas: HTMLCanvasElement } | null>(null);
  const sectionGroup = useMemo(() => new THREE.Group(), []);
  const boreholeGroup = useMemo(() => new THREE.Group(), []);
  const boreholesRef = useRef<THREE.Line[]>([]);

  // Data3D texture on field change.
  useEffect(() => {
    if (!manifest || !field) return;
    const tex = buildData3D(field, manifest);
    const prev = material.uniforms.uData.value as THREE.Data3DTexture | null;
    material.uniforms.uData.value = tex;
    prev?.dispose();
  }, [manifest, field, material]);

  // LUT texture on colormap change.
  useEffect(() => {
    const tex = buildLUTTexture(lut);
    const prev = material.uniforms.uLUT.value as THREE.DataTexture | null;
    material.uniforms.uLUT.value = tex;
    prev?.dispose();
  }, [lut, material]);

  // Uniforms: opacity + clipping plane.
  useEffect(() => {
    if (!grid) return;
    material.uniforms.uOpacity.value = volumeOpacity;
    material.uniforms.uClipOn.value = sectionOn ? 1 : 0;
    const axisNum = sectionAxis === "x" ? 0 : sectionAxis === "y" ? 1 : 2;
    material.uniforms.uClipAxis.value = axisNum;
    const dim = axisNum === 0 ? grid.nx : axisNum === 1 ? grid.ny : grid.nz;
    const frac = sectionIndex / (dim - 1);
    material.uniforms.uClipPos.value = axisNum === 2 ? 1 - frac : frac;
    material.uniforms.uClipSide.value = sectionReverse ? -1 : 1;
  }, [grid, volumeOpacity, sectionOn, sectionAxis, sectionIndex, sectionReverse, material]);

  // Section cut-face quad (create once).
  useEffect(() => {
    if (!manifest || !field || sectionRef.current) return;
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(18), 3));
    geom.setAttribute("uv", new THREE.BufferAttribute(
      new Float32Array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]), 2));
    const canvas = document.createElement("canvas");
    canvas.width = 2;
    canvas.height = 2; // valid initial size to avoid a 0-dim texture upload warning
    const tex = new THREE.CanvasTexture(canvas);
    const mat = new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(geom, mat);
    sectionGroup.add(mesh);
    sectionRef.current = { mesh, geom, tex, canvas };
    return () => {
      sectionGroup.remove(mesh);
      geom.dispose();
      mat.dispose();
      tex.dispose();
      sectionRef.current = null;
    };
  }, [manifest, field, sectionGroup]);

  // Update cut-face quad on section change.
  useEffect(() => {
    const s = sectionRef.current;
    if (!s || !manifest || !field) return;
    s.mesh.visible = sectionOn;
    if (!sectionOn) return;
    drawSlice(s.canvas, field, manifest, sectionAxis, sectionIndex, lut);
    s.tex.needsUpdate = true;
    writeQuad(s.geom, corners(manifest, sectionAxis, sectionIndex));
  }, [manifest, field, sectionOn, sectionAxis, sectionIndex, lut]);

  // Boreholes (create once).
  useEffect(() => {
    if (!manifest || !wells.length || boreholesRef.current.length) return;
    for (const w of wells) {
      const geom = new THREE.BufferGeometry();
      const pos: number[] = [];
      for (const smp of w.samples) pos.push(w.x * S, w.y * S, smp.z * S);
      geom.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
      geom.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(pos.length), 3));
      const line = new THREE.Line(geom, new THREE.LineBasicMaterial({ vertexColors: true }));
      boreholeGroup.add(line);
      boreholesRef.current.push(line);
    }
    return () => {
      for (const l of boreholesRef.current) {
        boreholeGroup.remove(l);
        l.geometry.dispose();
        (l.material as THREE.Material).dispose();
      }
      boreholesRef.current = [];
    };
  }, [manifest, wells, boreholeGroup]);

  // Borehole colors + visibility.
  useEffect(() => {
    if (!field) return;
    const span = field.meta.max - field.meta.min || 1;
    boreholesRef.current.forEach((line, wi) => {
      line.visible = showBoreholes;
      const col = line.geometry.getAttribute("color") as THREE.BufferAttribute;
      const arr = col.array as Float32Array;
      wells[wi].samples.forEach((smp, i) => {
        const tv = Math.min(1, Math.max(0, (smp.ucs_pred - field.meta.min) / span));
        const li = Math.round(tv * 255) * 3;
        arr[i * 3] = lut[li] / 255;
        arr[i * 3 + 1] = lut[li + 1] / 255;
        arr[i * 3 + 2] = lut[li + 2] / 255;
      });
      col.needsUpdate = true;
    });
  }, [field, lut, showBoreholes, wells]);

  if (!manifest) return null;
  const { x, y, z } = manifest.axes;
  const sx = (x[x.length - 1] - x[0]) * S;
  const sy = (y[y.length - 1] - y[0]) * S;
  const sz = (z[0] - z[z.length - 1]) * S; // 0 - (-120) = 120
  const cx = ((x[0] + x[x.length - 1]) / 2) * S;
  const cy = ((y[0] + y[y.length - 1]) / 2) * S;
  const cz = ((z[0] + z[z.length - 1]) / 2) * S;
  const box = new THREE.Box3(
    new THREE.Vector3(0, 0, z[z.length - 1] * S),
    new THREE.Vector3(x[x.length - 1] * S, y[y.length - 1] * S, 0),
  );

  return (
    <>
      <ambientLight intensity={1} />
      <mesh position={[cx, cy, cz]} scale={[sx, sy, sz]}>
        <boxGeometry args={[1, 1, 1]} />
        <primitive object={material} attach="material" />
      </mesh>
      <primitive object={sectionGroup} />
      <primitive object={boreholeGroup} />
      <box3Helper args={[box, new THREE.Color("#3a4152")]} />
      <axesHelper args={[0.8]} />
      <OrbitControls target={[cx, cy, cz]} makeDefault enableDamping dampingFactor={0.12} />
    </>
  );
}

export default function Volume3D() {
  const manifest = useStore((s) => s.manifest);
  const cam = useMemo<[number, number, number]>(() => {
    if (!manifest) return [6, -5, 6];
    const { x, y } = manifest.axes;
    return [x[x.length - 1] * S * 1.5, -y[y.length - 1] * S * 1.3, y[y.length - 1] * S * 1.7];
  }, [manifest]);
  if (!manifest) return null;
  return (
    <div className="view3d">
      <Canvas camera={{ position: cam, fov: 45, up: [0, 0, 1] }} dpr={[1, 2]}>
        <color attach="background" args={["#0b0e14"]} />
        <Scene />
      </Canvas>
    </div>
  );
}
