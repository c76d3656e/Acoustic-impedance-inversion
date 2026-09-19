import { useEffect, useMemo, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useShallow } from "zustand/react/shallow";
import * as THREE from "three";
import { useStore } from "../store";
import { buildLUT } from "../viz/colormaps";
import { drawScaled, extractSlice, sliceToImageData } from "../viz/slice";
import type { FieldVolume, Manifest, SliceAxis, Well } from "../types";

const S = 1 / 100; // meters -> world units
const AXES: SliceAxis[] = ["x", "y", "z"];
const ACTIVE_EDGE = new THREE.Color("#5ac8fa");
const IDLE_EDGE = new THREE.Color("#5a6478");

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

  const lut = useMemo(() => buildLUT(colormap, reverse), [colormap, reverse]);
  const planeGroup = useMemo(() => new THREE.Group(), []);
  const boreholeGroup = useMemo(() => new THREE.Group(), []);
  const planesRef = useRef<Plane[] | null>(null);
  const boreholesRef = useRef<THREE.Line[]>([]);

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
    for (const p of planes) {
      const idx = sliceIndex[p.axis];
      paintSlice(p.canvas, field, manifest, p.axis, idx, lut, wells, showBoreholes);
      p.tex.needsUpdate = true;
      const c = corners(manifest, p.axis, idx);
      writeQuad(p.geom, c);
      writeLoop(p.edgeGeom, c);
      const active = p.axis === axis;
      p.mat.opacity = 1;
      p.mat.transparent = false;
      p.mat.depthWrite = true;
      p.mesh.renderOrder = active ? 3 : 1;
      p.edgeMat.color.copy(active ? ACTIVE_EDGE : IDLE_EDGE);
    }
  }, [manifest, field, lut, axis, sliceIndex, wells, showBoreholes]);

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
  const cx = ((x[0] + x[x.length - 1]) / 2) * S;
  const cy = ((y[0] + y[y.length - 1]) / 2) * S;
  const cz = ((z[0] + z[z.length - 1]) / 2) * S;
  const box = new THREE.Box3(
    new THREE.Vector3(0, 0, z[z.length - 1] * S),
    new THREE.Vector3(x[x.length - 1] * S, y[y.length - 1] * S, 0),
  );

  return (
    <>
      <primitive object={planeGroup} />
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
    return [x[x.length - 1] * S * 1.55, -y[y.length - 1] * S * 1.35, y[y.length - 1] * S * 1.55];
  }, [manifest]);
  if (!manifest) return null;
  return (
    <div className="view3d">
      <Canvas camera={{ position: cam, fov: 45, up: [0, 0, 1] }} dpr={[1, 2]} gl={{ antialias: true }}>
        <color attach="background" args={["#0b0e14"]} />
        <Scene />
      </Canvas>
    </div>
  );
}
