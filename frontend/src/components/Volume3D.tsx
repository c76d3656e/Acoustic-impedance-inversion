import { useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { useStore } from "../store";
import { buildLUT } from "../viz/colormaps";
import { extractSlice, sliceToImageData } from "../viz/slice";
import type { FieldVolume, Manifest, SliceAxis } from "../types";

const S = 1 / 100; // meters -> world units

function sliceCanvas(
  field: FieldVolume,
  m: Manifest,
  axis: SliceAxis,
  index: number,
  lut: Uint8Array,
): HTMLCanvasElement {
  const slice = extractSlice(field, m, axis, index);
  const img = sliceToImageData(slice, lut, field.meta.min, field.meta.max);
  const cv = document.createElement("canvas");
  cv.width = slice.w;
  cv.height = slice.h;
  cv.getContext("2d")!.putImageData(img, 0, 0);
  return cv;
}

function quadMesh(
  corners: [THREE.Vector3, THREE.Vector3, THREE.Vector3, THREE.Vector3],
  tex: THREE.Texture,
): THREE.Mesh {
  const g = new THREE.BufferGeometry();
  const [bl, br, tr, tl] = corners;
  const pos = new Float32Array([
    bl.x, bl.y, bl.z, br.x, br.y, br.z, tr.x, tr.y, tr.z,
    bl.x, bl.y, bl.z, tr.x, tr.y, tr.z, tl.x, tl.y, tl.z,
  ]);
  const uv = new Float32Array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]);
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  g.computeVertexNormals();
  const mat = new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide });
  return new THREE.Mesh(g, mat);
}

function Scene() {
  const { manifest, sliceIndex, colormap, reverse, showSlices, showBoreholes, wells } =
    useStore();
  const field = useStore((s) => s.currentField());
  const lut = useMemo(() => buildLUT(colormap, reverse), [colormap, reverse]);

  const planes = useMemo(() => {
    if (!manifest || !field || !showSlices) return [];
    const { x, y, z } = manifest.axes;
    const xw = (v: number) => v * S, yw = (v: number) => v * S, zw = (v: number) => v * S;
    const out: THREE.Mesh[] = [];

    // Z plane (horizontal) at selected elevation
    {
      const zc = zw(z[sliceIndex.z]);
      const tex = new THREE.CanvasTexture(sliceCanvas(field, manifest, "z", sliceIndex.z, lut));
      out.push(quadMesh([
        new THREE.Vector3(xw(x[0]), yw(y[0]), zc),
        new THREE.Vector3(xw(x[x.length - 1]), yw(y[0]), zc),
        new THREE.Vector3(xw(x[x.length - 1]), yw(y[y.length - 1]), zc),
        new THREE.Vector3(xw(x[0]), yw(y[y.length - 1]), zc),
      ], tex));
    }
    // X plane (YZ) at selected X
    {
      const xc = xw(x[sliceIndex.x]);
      const tex = new THREE.CanvasTexture(sliceCanvas(field, manifest, "x", sliceIndex.x, lut));
      out.push(quadMesh([
        new THREE.Vector3(xc, yw(y[0]), zw(z[z.length - 1])),
        new THREE.Vector3(xc, yw(y[y.length - 1]), zw(z[z.length - 1])),
        new THREE.Vector3(xc, yw(y[y.length - 1]), zw(z[0])),
        new THREE.Vector3(xc, yw(y[0]), zw(z[0])),
      ], tex));
    }
    // Y plane (XZ) at selected Y
    {
      const yc = yw(y[sliceIndex.y]);
      const tex = new THREE.CanvasTexture(sliceCanvas(field, manifest, "y", sliceIndex.y, lut));
      out.push(quadMesh([
        new THREE.Vector3(xw(x[0]), yc, zw(z[z.length - 1])),
        new THREE.Vector3(xw(x[x.length - 1]), yc, zw(z[z.length - 1])),
        new THREE.Vector3(xw(x[x.length - 1]), yc, zw(z[0])),
        new THREE.Vector3(xw(x[0]), yc, zw(z[0])),
      ], tex));
    }
    return out;
  }, [manifest, field, sliceIndex, lut, showSlices]);

  const boreholes = useMemo(() => {
    if (!manifest || !field || !showBoreholes) return [];
    const span = field.meta.max - field.meta.min || 1;
    return wells.map((w) => {
      const g = new THREE.BufferGeometry();
      const pos: number[] = [];
      const col: number[] = [];
      for (const s of w.samples) {
        pos.push(w.x * S, w.y * S, s.z * S);
        const tv = Math.min(1, Math.max(0, (s.ucs_pred - field.meta.min) / span));
        const li = Math.round(tv * 255) * 3;
        col.push(lut[li] / 255, lut[li + 1] / 255, lut[li + 2] / 255);
      }
      g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
      g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
      return new THREE.Line(g, new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 2 }));
    });
  }, [manifest, field, wells, showBoreholes, lut]);

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
      <group>
        {planes.map((mesh, i) => <primitive key={`p${i}`} object={mesh} />)}
        {boreholes.map((line, i) => <primitive key={`b${i}`} object={line} />)}
        <box3Helper args={[box, new THREE.Color("#888")]} />
      </group>
      <axesHelper args={[1]} />
      <OrbitControls target={[cx, cy, cz]} makeDefault />
    </>
  );
}

export default function Volume3D() {
  const { manifest } = useStore();
  if (!manifest) return null;
  const { x, y } = manifest.axes;
  const cam: [number, number, number] = [
    x[x.length - 1] * S * 1.4, -y[y.length - 1] * S * 1.2, y[y.length - 1] * S * 1.6,
  ];
  return (
    <div className="view3d">
      <Canvas camera={{ position: cam, fov: 45, up: [0, 0, 1] }}>
        <color attach="background" args={["#0e1116"]} />
        <Scene />
      </Canvas>
    </div>
  );
}
