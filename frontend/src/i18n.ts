// Full Chinese localization. All user-facing text lives here so the CJK font
// subset can be generated from a single source of truth.
export const t = {
  appTitle: "露天矿波阻抗与岩石强度三维可视化",
  subtitle: "无后端 · 纯前端 · 本地计算（WebGL2 + Pyodide/matplotlib）",
  loading: "正在加载数据集…",
  loadingPyodide: "正在加载 Pyodide 与 matplotlib（首次较慢，约 10–30 秒）…",
  loadingCompare: "正在加载对比场…",

  panelField: "数据场",
  panelSlice: "切片控制",
  panelColormap: "配色方案",
  panelWells: "钻孔（多井）",
  panelExport: "出图（本地 matplotlib）",
  panelView: "视图",

  axis: "切片方向",
  axisX: "X 切片（YZ 面）",
  axisY: "Y 切片（XZ 面）",
  axisZ: "Z 切片（水平面/标高）",
  slicePos: "切片位置",
  elevation: "标高",
  meter: "米",

  colormapPreset: "预设",
  colormapCustom: "自定义强度配色",
  addStop: "添加色标",
  removeStop: "删除",
  reverse: "反转",

  view2d: "二维切片",
  view3d: "三维视图",
  viewCompare: "融合对比",
  showBoreholes: "显示钻孔",
  sliceWysiwygHint: "三维、二维与出图共用这一切片，移动滑条即所见即所得。",

  exportKind: "出图类型",
  exportKindSlice: "当前切片",
  exportKindCompare: "融合对比",
  exportKindProfile: "沿孔剖面",
  exportButton: "生成出版级图片（Nature 风格）",
  exporting: "正在本地渲染…",
  download: "下载 PNG",
  natureStyle: "Nature 风格优化",
  exportHint: "根据当前所选数据场、切片位置与配色，在本地生成带中文标注的高分辨率图片。",
  exportHintCompare: "与文档相同的 2×4 图：真值 / 仅钻孔 / 仅波阻抗 / 融合，下行预测减真值热力图。",
  exportHintProfile: "沿孔剖面：真值、仅钻孔、仅波阻抗标定与外漂移克里金融合（孔上钉回硬数据）。",

  wellSelect: "选择钻孔",
  wellLogTitle: "沿孔测井曲线",
  wellStrength: "孔径强度（UCS）",
  depth: "深度/标高 (m)",
  wellPoints: "钻孔点位",
  wellCount: "钻孔数量",

  colorbar: "数值",
  min: "最小",
  max: "最大",
  value: "取值",

  curveV: "钻速 V",
  curveN: "转速 N",
  curveM: "扭矩 M",
  curveF: "钻压 F",
  curveUCSpred: "MWD",
  curveUCStrue: "真实UCS",
  curveUCSseis: "波阻抗UCS",
  curveUCSfused: "融合UCS",

  residualNote: "下行：预测 − 真值\n红＝估计偏高\n蓝＝估计偏低\n越浅越好",
  residualCbar: "预测 − 真值 (MPa)",
  compareSuptitle: "融合优势对比",
  profileSuptitle: "沿孔剖面：融合在孔上钉回硬数据；波阻抗只给出趋势",
  clickPanel: "点击面板可切换到该场的二维切片",
};

export type I18nKey = keyof typeof t;
