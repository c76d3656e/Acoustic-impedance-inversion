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

function corners(
  m: Manifest,
  axis: SliceAxis,
  index: number,
): [THREE.Vector3, THREE.Vector3, THREE.Vector3, THREE.Vector3] {
  const { x, y, z } = m.axes;
  const X = (v: number) => v * S, Y = (v: number) => v * S, Z = (v: number) => v * S;
  const xN = x.length - 1, yN = y.length - 1, zN = z.length - 1;
  if (axis === "z") {
    const zc = Z(z[index]);
    return [
      new THREE.Vector3(X(x[0]), Y(y[0]), zc),
      new THREE.Vector3(X(x[xN]), Y(y[0]), zc),
      new THREE.Vector3(X(x[xN]), Y(y[yN]), zc),
      new THREE.Vector3(X(x[0]), Y(y[yN]), zc),
    ];
  }
  if (axis === "x") {
    const xc = X(x[index]);
    return [
      new THREE.Vector3(xc, Y(y[0]), Z(z[zN])),
      new THREE.Vector3(xc, Y(y[yN]), Z(z[zN])),
      new THREE.Vector3(xc, Y(y[yN]), Z(z[0])),
      new THREE.Vector3(xc, Y(y[0]), Z(z[0])),
    ];
  }
  const yc = Y(y[index]);
  return [
    new THREE.Vector3(X(x[0]), yc, Z(z[zN])),
    new THREE.Vector3(X(x[xN]), yc, Z(z[zN])),
    new THREE.Vector3(X(x[xN]), yc, Z(z[0])),
    new THREE.Vector3(X(x[0]), yc, Z(z[0])),
  ];
}

function writeQuad(geom: THREE.BufferGeometry, c: ReturnType<typeof corners>) {
  const [bl, br, tr, tl] = c;
  const p = geom.getAttribute("position") as THREE.BufferAttribute;
  const a = p.array as Float32Array;
  a.set([
    bl.x, bl.y, bl.z, br.x, br.y, br.z, tr.x, tr.y, tr.z,
    bl.x, bl.y, bl.z, tr.x, tr.y, tr.z, tl.x, tl.y, tl.z,
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

interface Plane {
  axis: SliceAxis;
  mesh: THREE.Mesh;
  geom: THREE.BufferGeometry;
  tex: THREE.CanvasTexture;
  canvas: HTMLCanvasElement;
}

function Scene() {
  const manifest = useStore((s) => s.manifest);
  const field = useStore((s) => s.currentField());
  const wells = useStore((s) => s.wells);
  const sliceIndex = useStore(useShallow((s) => s.sliceIndex));
  const colormap = useStore((s) => s.colormap);
  const reverse = useStore((s) => s.reverse);
  const showSlices = useStore((s) => s.showSlices);
  const showBoreholes = useStore((s) => s.showBoreholes);

  const lut = useMemo(() => buildLUT(colormap, reverse), [colormap, reverse]);
  const group = useMemo(() => new THREE.Group(), []);
  const planesRef = useRef<Plane[]>([]);
  const boreholesRef = useRef<THREE.Line[]>([]);

  // Create persistent slice planes once.
  useEffect(() => {
    if (!manifest || !field || planesRef.current.length) return;
    const axes: SliceAxis[] = ["z", "x", "y"];
    for (const axis of axes) {
      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(18), 3));
      geom.setAttribute(
        "uv",
        new THREE.BufferAttribute(
          new Float32Array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]), 2,
        ),
      );
      const canvas = document.createElement("canvas");
      const tex = new THREE.CanvasTexture(canvas);
      const mat = new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide });
      const mesh = new THREE.Mesh(geom, mat);
      group.add(mesh);
      planesRef.current.push({ axis, mesh, geom, tex, canvas });
    }
    return () => {
      for (const p of planesRef.current) {
        group.remove(p.mesh);
        p.geom.dispose();
        (p.mesh.material as THREE.Material).dispose();
        p.tex.dispose();
      }
      planesRef.current = [];
    };
  }, [manifest, field, group]);

  // Create persistent borehole lines once (positions are static).
  useEffect(() => {
    if (!manifest || !wells.length || boreholesRef.current.length) return;
    for (const w of wells) {
      const geom = new THREE.BufferGeometry();
      const pos: number[] = [];
      for (const s of w.samples) pos.push(w.x * S, w.y * S, s.z * S);
      geom.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
      geom.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(pos.length), 3));
      const line = new THREE.Line(geom, new THREE.LineBasicMaterial({ vertexColors: true }));
      group.add(line);
      boreholesRef.current.push(line);
    }
    return () => {
      for (const l of boreholesRef.current) {
        group.remove(l);
        l.geometry.dispose();
        (l.material as THREE.Material).dispose();
      }
      boreholesRef.current = [];
    };
  }, [manifest, wells, group]);

  // Update slice textures + geometry on change (mutate, don't recreate).
  useEffect(() => {
    if (!manifest || !field) return;
    for (const p of planesRef.current) {
      p.mesh.visible = showSlices;
      if (!showSlices) continue;
      drawSlice(p.canvas, field, manifest, p.axis, sliceIndex[p.axis], lut);
      p.tex.needsUpdate = true;
      writeQuad(p.geom, corners(manifest, p.axis, sliceIndex[p.axis]));
    }
  }, [manifest, field, sliceIndex, lut, showSlices]);

  // Update borehole colors on field/lut change; toggle visibility.
  useEffect(() => {
    if (!field) return;
    const span = field.meta.max - field.meta.min || 1;
    boreholesRef.current.forEach((line, wi) => {
      line.visible = showBoreholes;
      const w = wells[wi];
      const col = line.geometry.getAttribute("color") as THREE.BufferAttribute;
      const arr = col.array as Float32Array;
      w.samples.forEach((s, i) => {
        const tv = Math.min(1, Math.max(0, (s.ucs_pred - field.meta.min) / span));
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
  const cx = (x[x.length - 1] * S) / 2, cy = (y[y.length - 1] * S) / 2, cz = (z[z.length - 1] * S) / 2;
  const box = new THREE.Box3(
    new THREE.Vector3(0, 0, z[z.length - 1] * S),
    new THREE.Vector3(x[x.length - 1] * S, y[y.length - 1] * S, 0),
  );

  return (
    <>
      <ambientLight intensity={1} />
      <primitive object={group} />
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
    return [x[x.length - 1] * S * 1.4, -y[y.length - 1] * S * 1.2, y[y.length - 1] * S * 1.6];
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
