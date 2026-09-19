import { Component, useEffect, useMemo, useRef, type ReactNode } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useShallow } from "zustand/react/shallow";
import * as THREE from "three";
import { useStore } from "../store";
import { buildLUT } from "../viz/colormaps";
import { drawScaled, extractSlice, sliceToImageData } from "../viz/slice";
import type { FieldVolume, Manifest, SliceAxis, Well } from "../types";
import { cropWindow, type ViewCrop } from "../viz/window";
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

function corners(m: Manifest, axis: SliceAxis, index: number, crop: ViewCrop) {
  const { x, y, z } = m.axes;
  const X = (v: number) => v * S, Y = (v: number) => v * S, Z = (v: number) => v * S;
  const x0 = crop.x0, x1 = crop.x1, y0 = crop.y0, y1 = crop.y1;
  const zN = z.length - 1;
  if (axis === "z") {
    const zc = Z(z[index]);
    return [
      [X(x0), Y(y0), zc], [X(x1), Y(y0), zc],
      [X(x1), Y(y1), zc], [X(x0), Y(y1), zc],
    ];
  }
  if (axis === "x") {
    const xc = X(x[index]);
    return [
      [xc, Y(y0), Z(z[zN])], [xc, Y(y1), Z(z[zN])],
      [xc, Y(y1), Z(z[0])], [xc, Y(y0), Z(z[0])],
    ];
  }
  const yc = Y(y[index]);
  return [
    [X(x0), yc, Z(z[zN])], [X(x1), yc, Z(z[zN])],
    [X(x1), yc, Z(z[0])], [X(x0), yc, Z(z[0])],
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
  crop: ViewCrop,
) {
  const slice = extractSlice(field, m, axis, index, crop);
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

function buildData3D(field: FieldVolume, m: Manifest, crop: ViewCrop): THREE.Data3DTexture {
  const { ny, nz } = m.grid;
  const nxC = crop.ix1 - crop.ix0 + 1;
  const nyC = crop.iy1 - crop.iy0 + 1;
  const { min, max } = field.meta;
  const span = max - min || 1;
  const arr = new Uint8Array(nxC * nyC * nz);
  const d = field.data;
  for (let ix = crop.ix0; ix <= crop.ix1; ix++)
    for (let iy = crop.iy0; iy <= crop.iy1; iy++)
      for (let iz = 0; iz < nz; iz++) {
        const v = d[(ix * ny + iy) * nz + iz];
        const jx = ix - crop.ix0, jy = iy - crop.iy0;
        arr[jx + jy * nxC + iz * nxC * nyC] = Math.max(0, Math.min(255,
          Math.round(((v - min) / span) * 255)));
      }
  const tex = new THREE.Data3DTexture(arr, nxC, nyC, nz);
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
  const winX0 = useStore((s) => s.winX0);
  const winY0 = useStore((s) => s.winY0);
  const theme = useStore((s) => s.theme);
  const grid = useStore(useShallow((s) => s.manifest?.grid));

  const lut = useMemo(() => buildLUT(colormap, reverse), [colormap, reverse]);
  const crop = useMemo(
    () => (manifest ? cropWindow(manifest, winX0, winY0) : null),
    [manifest, winX0, winY0],
  );
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
    if (!manifest || !field || !crop) return;
    const tex = buildData3D(field, manifest, crop);
    const prev = material.uniforms.uData.value as THREE.Data3DTexture | null;
    material.uniforms.uData.value = tex;
    prev?.dispose();
  }, [manifest, field, material, crop]);

  useEffect(() => {
    const tex = buildLUTTexture(lut);
    const prev = material.uniforms.uLUT.value as THREE.DataTexture | null;
    material.uniforms.uLUT.value = tex;
    prev?.dispose();
  }, [lut, material]);

  useEffect(() => {
    if (!grid || !crop) return;
    const voxel = volumeStyle === "voxel";
    material.uniforms.uOpacity.value = volumeOpacity;
    material.uniforms.uClipOn.value = voxel ? 1 : 0;
    const axisNum = axis === "x" ? 0 : axis === "y" ? 1 : 2;
    material.uniforms.uClipAxis.value = axisNum;
    const dim = axisNum === 0
      ? (crop.ix1 - crop.ix0 + 1)
      : axisNum === 1
        ? (crop.iy1 - crop.iy0 + 1)
        : grid.nz;
    const local = axis === "x"
      ? sliceIndex.x - crop.ix0
      : axis === "y"
        ? sliceIndex.y - crop.iy0
        : sliceIndex.z;
    const frac = local / Math.max(1, dim - 1);
    material.uniforms.uClipPos.value = axisNum === 2 ? 1 - frac : frac;
    material.uniforms.uClipSide.value = sectionReverse ? -1 : 1;
  }, [grid, crop, volumeStyle, volumeOpacity, axis, sliceIndex, sectionReverse, material]);

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
    if (!planes || !manifest || !field || !crop) return;
    const slices = volumeStyle === "slices";
    const idle = theme === "light" ? new THREE.Color("#8a9099") : IDLE_EDGE;
    for (const p of planes) {
      const idx = sliceIndex[p.axis];
      const active = p.axis === axis;
      const show = slices || active;
      p.mesh.visible = show;
      p.edge.visible = show;
      if (!show) continue;
      paintSlice(p.canvas, field, manifest, p.axis, idx, lut, wells, showBoreholes, crop);
      p.tex.needsUpdate = true;
      const c = corners(manifest, p.axis, idx, crop);
      writeQuad(p.geom, c);
      writeLoop(p.edgeGeom, c);
      p.mesh.renderOrder = active ? 3 : 1;
      p.edgeMat.color.copy(active ? ACTIVE_EDGE : idle);
    }
  }, [manifest, field, lut, axis, sliceIndex, wells, showBoreholes, volumeStyle, crop, theme]);

  useEffect(() => {
    if (!manifest || !wells.length || !crop) return;
    for (const l of boreholesRef.current) {
      boreholeGroup.remove(l);
      l.geometry.dispose();
      (l.material as THREE.Material).dispose();
    }
    boreholesRef.current = [];
    for (const w of wells) {
      if (w.x < crop.x0 || w.x > crop.x1 || w.y < crop.y0 || w.y > crop.y1) continue;
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
  }, [manifest, wells, boreholeGroup, crop]);

  useEffect(() => {
    if (!field || !crop) return;
    const span = field.meta.max - field.meta.min || 1;
    const visible = wells.filter(
      (w) => w.x >= crop.x0 && w.x <= crop.x1 && w.y >= crop.y0 && w.y <= crop.y1,
    );
    boreholesRef.current.forEach((line, wi) => {
      line.visible = showBoreholes;
      const well = visible[wi];
      if (!well) return;
      const col = line.geometry.getAttribute("color") as THREE.BufferAttribute;
      const arr = col.array as Float32Array;
      well.samples.forEach((smp, i) => {
        const tv = Math.min(1, Math.max(0, (smp.ucs_pred - field.meta.min) / span));
        const li = Math.round(tv * 255) * 3;
        arr[i * 3] = lut[li] / 255;
        arr[i * 3 + 1] = lut[li + 1] / 255;
        arr[i * 3 + 2] = lut[li + 2] / 255;
      });
      col.needsUpdate = true;
    });
  }, [field, lut, showBoreholes, wells, crop]);

  if (!manifest || !crop) return null;
  const { z } = manifest.axes;
  const sx = (crop.x1 - crop.x0) * S;
  const sy = (crop.y1 - crop.y0) * S;
  const sz = (z[0] - z[z.length - 1]) * S;
  const cx = ((crop.x0 + crop.x1) / 2) * S;
  const cy = ((crop.y0 + crop.y1) / 2) * S;
  const cz = ((z[0] + z[z.length - 1]) / 2) * S;
  const box = new THREE.Box3(
    new THREE.Vector3(crop.x0 * S, crop.y0 * S, z[z.length - 1] * S),
    new THREE.Vector3(crop.x1 * S, crop.y1 * S, 0),
  );
  const boxColor = theme === "light" ? "#b4b8bf" : "#3a4152";

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
      <box3Helper args={[box, new THREE.Color(boxColor)]} />
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
  const theme = useStore((s) => s.theme);
  const winX0 = useStore((s) => s.winX0);
  const winY0 = useStore((s) => s.winY0);
  const cam = useMemo<[number, number, number]>(() => {
    if (!manifest) return [6, -5, 6];
    const crop = cropWindow(manifest, winX0, winY0);
    return [crop.x1 * S * 1.55, -crop.y1 * S * 1.35, crop.y1 * S * 1.55];
  }, [manifest, winX0, winY0]);
  if (!manifest) return null;
  const bg = theme === "light" ? "#ffffff" : "#0b0e14";
  return (
    <div className="view3d">
      <GLErrorBoundary>
        <Canvas
          camera={{ position: cam, fov: 45, up: [0, 0, 1] }}
          dpr={[1, 2]}
          gl={{ antialias: true, failIfMajorPerformanceCaveat: false, powerPreference: "high-performance" }}
        >
          <color attach="background" args={[bg]} />
          <Scene />
        </Canvas>
      </GLErrorBoundary>
    </div>
  );
}
