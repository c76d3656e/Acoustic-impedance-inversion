import { Component, useEffect, useMemo, useRef, type ReactNode } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useShallow } from "zustand/react/shallow";
import * as THREE from "three";
import { useStore } from "../store";
import { buildLUT } from "../viz/colormaps";
import { drawScaled, extractSlice, sliceToImageData } from "../viz/slice";
import type { FieldVolume, Manifest, SliceAxis, Well } from "../types";
import { t } from "../i18n";

const S = 1 / 100; // meters -> world units
const AXES: SliceAxis[] = ["x", "y", "z"];
const ACTIVE_EDGE = new THREE.Color("#5ac8fa");
const IDLE_EDGE = new THREE.Color("#5a6478");

const VERT = /* glsl */ `
out vec3 vOrigin;
out vec3 vDir;
void main() {
  vec3 p = position + 0.5;
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

function writeLoop(geom: THREE.BufferGeometry, c: number[][]) {
  const [bl, br, tr, tl] = c;
  const p = geom.getAttribute("position") as THREE.BufferAttribute;
  (p.array as Float32Array).set([...bl, ...br, ...tr, ...tl]);
  p.needsUpdate = true;
  geom.computeBoundingSphere();
}

function paintSlice(
  canvas: HTMLCanvasElement,
  field: FieldVolume,
  m: Manifest,
  axis: SliceAxis,
  index: number,
  lut: Uint8Array,
  wells: Well[],
  showHoles: boolean,
) {
  const slice = extractSlice(field, m, axis, index);
  const img = sliceToImageData(slice, lut, field.meta.min, field.meta.max);
  const scale = Math.max(2, Math.round(512 / Math.max(slice.w, slice.h)));
  canvas.width = slice.w * scale;
  canvas.height = slice.h * scale;
  drawScaled(canvas, img);
  if (showHoles && axis === "z") {
    const ctx = canvas.getContext("2d")!;
    const W = canvas.width, H = canvas.height;
    const [x0, x1, y0, y1] = slice.extent;
    for (const w of wells) {
      const px = ((w.x - x0) / (x1 - x0)) * W;
      const py = (1 - (w.y - y0) / (y1 - y0)) * H;
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, Math.PI * 2);
      ctx.strokeStyle = "#111";
      ctx.lineWidth = 1.6;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(px, py, 1.6, 0, Math.PI * 2);
      ctx.fillStyle = "#111";
      ctx.fill();
    }
  }
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

type Plane = {
  axis: SliceAxis;
  mesh: THREE.Mesh;
  edge: THREE.LineLoop;
  geom: THREE.BufferGeometry;
  edgeGeom: THREE.BufferGeometry;
  tex: THREE.CanvasTexture;
  canvas: HTMLCanvasElement;
  mat: THREE.MeshBasicMaterial;
  edgeMat: THREE.LineBasicMaterial;
};

function makePlane(axis: SliceAxis): Plane {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(18), 3));
  geom.setAttribute("uv", new THREE.BufferAttribute(
    new Float32Array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]), 2));
  const canvas = document.createElement("canvas");
  canvas.width = 2;
  canvas.height = 2;
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.flipY = true;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  const mat = new THREE.MeshBasicMaterial({
    map: tex,
    side: THREE.DoubleSide,
    transparent: false,
    opacity: 1,
    depthWrite: true,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });
  const mesh = new THREE.Mesh(geom, mat);
  const edgeGeom = new THREE.BufferGeometry();
  edgeGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(12), 3));
  const edgeMat = new THREE.LineBasicMaterial({ color: IDLE_EDGE });
  const edge = new THREE.LineLoop(edgeGeom, edgeMat);
  edge.renderOrder = 4;
  return { axis, mesh, edge, geom, edgeGeom, tex, canvas, mat, edgeMat };
}

function Scene() {
  const manifest = useStore((s) => s.manifest);
  const field = useStore((s) => s.currentField());
  const wells = useStore((s) => s.wells);
  const colormap = useStore((s) => s.colormap);
  const reverse = useStore((s) => s.reverse);
  const showBoreholes = useStore((s) => s.showBoreholes);
  const axis = useStore((s) => s.axis);
  const sliceIndex = useStore(useShallow((s) => s.sliceIndex));
  const volumeStyle = useStore((s) => s.volumeStyle);
  const volumeOpacity = useStore((s) => s.volumeOpacity);
  const sectionReverse = useStore((s) => s.sectionReverse);
  const grid = useStore(useShallow((s) => s.manifest?.grid));

  const lut = useMemo(() => buildLUT(colormap, reverse), [colormap, reverse]);
  const planeGroup = useMemo(() => new THREE.Group(), []);
  const boreholeGroup = useMemo(() => new THREE.Group(), []);
  const planesRef = useRef<Plane[] | null>(null);
  const boreholesRef = useRef<THREE.Line[]>([]);

  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        glslVersion: THREE.GLSL3,
        uniforms: {
          uData: { value: null },
          uLUT: { value: null },
          uOpacity: { value: 0.45 },
          uSteps: { value: 160 },
          uClipOn: { value: 1 },
          uClipAxis: { value: 2 },
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

  useEffect(() => {
    if (!manifest || !field) return;
    const tex = buildData3D(field, manifest);
    const prev = material.uniforms.uData.value as THREE.Data3DTexture | null;
    material.uniforms.uData.value = tex;
    prev?.dispose();
  }, [manifest, field, material]);

  useEffect(() => {
    const tex = buildLUTTexture(lut);
    const prev = material.uniforms.uLUT.value as THREE.DataTexture | null;
    material.uniforms.uLUT.value = tex;
    prev?.dispose();
  }, [lut, material]);

  useEffect(() => {
    if (!grid) return;
    const voxel = volumeStyle === "voxel";
    material.uniforms.uOpacity.value = volumeOpacity;
    material.uniforms.uClipOn.value = voxel ? 1 : 0;
    const axisNum = axis === "x" ? 0 : axis === "y" ? 1 : 2;
    material.uniforms.uClipAxis.value = axisNum;
    const dim = axisNum === 0 ? grid.nx : axisNum === 1 ? grid.ny : grid.nz;
    const frac = sliceIndex[axis] / Math.max(1, dim - 1);
    material.uniforms.uClipPos.value = axisNum === 2 ? 1 - frac : frac;
    material.uniforms.uClipSide.value = sectionReverse ? -1 : 1;
  }, [grid, volumeStyle, volumeOpacity, axis, sliceIndex, sectionReverse, material]);

  useEffect(() => () => {
    (material.uniforms.uData.value as THREE.Texture | null)?.dispose();
    (material.uniforms.uLUT.value as THREE.Texture | null)?.dispose();
    material.dispose();
  }, [material]);

  useEffect(() => {
    const planes = AXES.map(makePlane);
    for (const p of planes) {
      planeGroup.add(p.mesh);
      planeGroup.add(p.edge);
    }
    planesRef.current = planes;
    return () => {
      for (const p of planes) {
        planeGroup.remove(p.mesh);
        planeGroup.remove(p.edge);
        p.geom.dispose();
        p.edgeGeom.dispose();
        p.mat.dispose();
        p.edgeMat.dispose();
        p.tex.dispose();
      }
      planesRef.current = null;
    };
  }, [planeGroup]);

  useEffect(() => {
    const planes = planesRef.current;
    if (!planes || !manifest || !field) return;
    const slices = volumeStyle === "slices";
    for (const p of planes) {
      const idx = sliceIndex[p.axis];
      const active = p.axis === axis;
      const show = slices || active;
      p.mesh.visible = show;
      p.edge.visible = show;
      if (!show) continue;
      paintSlice(p.canvas, field, manifest, p.axis, idx, lut, wells, showBoreholes);
      p.tex.needsUpdate = true;
      const c = corners(manifest, p.axis, idx);
      writeQuad(p.geom, c);
      writeLoop(p.edgeGeom, c);
      p.mesh.renderOrder = active ? 3 : 1;
      p.edgeMat.color.copy(active ? ACTIVE_EDGE : IDLE_EDGE);
    }
  }, [manifest, field, lut, axis, sliceIndex, wells, showBoreholes, volumeStyle]);

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
  const sz = (z[0] - z[z.length - 1]) * S;
  const cx = ((x[0] + x[x.length - 1]) / 2) * S;
  const cy = ((y[0] + y[y.length - 1]) / 2) * S;
  const cz = ((z[0] + z[z.length - 1]) / 2) * S;
  const box = new THREE.Box3(
    new THREE.Vector3(0, 0, z[z.length - 1] * S),
    new THREE.Vector3(x[x.length - 1] * S, y[y.length - 1] * S, 0),
  );

  return (
    <>
      <mesh
        position={[cx, cy, cz]}
        scale={[sx, sy, sz]}
        visible={volumeStyle === "voxel"}
        renderOrder={0}
      >
        <boxGeometry args={[1, 1, 1]} />
        <primitive object={material} attach="material" />
      </mesh>
      <primitive object={planeGroup} />
      <primitive object={boreholeGroup} />
      <box3Helper args={[box, new THREE.Color("#3a4152")]} />
      <axesHelper args={[0.8]} />
      <OrbitControls target={[cx, cy, cz]} makeDefault enableDamping dampingFactor={0.12} />
    </>
  );
}

class GLErrorBoundary extends Component<{ children: ReactNode }, { err: string | null }> {
  state: { err: string | null } = { err: null };
  static getDerivedStateFromError(e: Error) {
    return { err: e.message || "WebGL" };
  }
  render() {
    if (this.state.err) return <div className="loading3d">{t.webglError}</div>;
    return this.props.children;
  }
}

export default function Volume3D() {
  const manifest = useStore((s) => s.manifest);
  const cam = useMemo<[number, number, number]>(() => {
    if (!manifest) return [6, -5, 6];
    const { x, y } = manifest.axes;
    return [x[x.length - 1] * S * 1.55, -y[y.length - 1] * S * 1.35, y[y.length - 1] * S * 1.55];
  }, [manifest]);
  if (!manifest) return null;
  return (
    <div className="view3d">
      <GLErrorBoundary>
        <Canvas
          camera={{ position: cam, fov: 45, up: [0, 0, 1] }}
          dpr={[1, 2]}
          gl={{ antialias: true, failIfMajorPerformanceCaveat: false, powerPreference: "high-performance" }}
        >
          <color attach="background" args={["#0b0e14"]} />
          <Scene />
        </Canvas>
      </GLErrorBoundary>
    </div>
  );
}
