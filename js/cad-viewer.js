import * as THREE from '../vendor/three/three.module.min.js';
import { OrbitControls } from '../vendor/three/OrbitControls.js';
import { STLLoader } from '../vendor/three/STLLoader.js';
import {
  MM,
  computeCreasedNormals,
  createAnatomyMaterials,
  createM2SocketBolt,
  createDualShearBushing,
  createServoHornAssembly,
  createPrecisionServo,
  createXiaoESP32S3SenseAssembly,
  createVL53L0XAssembly,
  createPCA9685Assembly,
  createMPU6050CoMAssembly,
  createDualBuckPowerNodes,
  createAudioSystemNodes,
  createBattery2SAndRearIO,
  createCompleteWiringHarnesses,
  attachKinematicAndCoMGuides,
} from './rnv1-anatomy.js';

// --- Setup Pure White (#FFFFFF) Studio WebGL Renderer with Selective Section Clipping ---
const canvas = document.getElementById('cad-canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0xffffff, 1);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.localClippingEnabled = true;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.06;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);

// --- Studio Three-Point + Interior Fill CAD Lighting ---
const hemiLight = new THREE.HemisphereLight(0xffffff, 0xe8edf5, 1.08);
hemiLight.position.set(0, 1.0, 0);
scene.add(hemiLight);

const keyLight = new THREE.DirectionalLight(0xffffff, 1.35);
keyLight.position.set(0.55, 0.95, 0.72);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
keyLight.shadow.camera.near = 0.1;
keyLight.shadow.camera.far = 3.0;
const sc = 0.38;
keyLight.shadow.camera.left = -sc;
keyLight.shadow.camera.right = sc;
keyLight.shadow.camera.top = sc;
keyLight.shadow.camera.bottom = -sc;
keyLight.shadow.bias = -0.0004;
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0xf2f6ff, 0.62);
fillLight.position.set(-0.65, 0.45, -0.45);
scene.add(fillLight);

const rimLight = new THREE.DirectionalLight(0xffffff, 0.52);
rimLight.position.set(-0.20, 0.65, 0.70);
scene.add(rimLight);

// Invisible Shadow-Catcher Plane on Pure White Studio Floor (Y = 0)
const shadowFloor = new THREE.Mesh(
  new THREE.PlaneGeometry(2.6, 2.6),
  new THREE.ShadowMaterial({ opacity: 0.13 })
);
shadowFloor.rotation.x = -Math.PI / 2;
shadowFloor.position.y = 0;
shadowFloor.receiveShadow = true;
scene.add(shadowFloor);

// --- Camera & Damped OrbitControls ---
const camera = new THREE.PerspectiveCamera(34, window.innerWidth / window.innerHeight, 0.008, 20);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.075;
controls.minDistance = 0.11;
controls.maxDistance = 1.35;
controls.maxPolarAngle = Math.PI * 0.495;

// Note: OpenSCAD (+X fwd, +Y left, +Z up) mapped via worldRobot.rotation.x = -PI/2 -> Three.js (+X fwd, +Y up, -Z left / +Z right)
const VIEW_PRESETS = [
  { pos: [0.38, 0.23, 0.38],  target: [0.0, 0.118, 0.0] },    // 0: 3/4 Isometric Hero
  { pos: [0.46, 0.13, 0.00],  target: [0.0, 0.118, 0.0] },    // 1: Front Elevation
  { pos: [0.02, 0.135, 0.46], target: [0.0, 0.118, 0.0] },    // 2: Right Sagittal Profile
  { pos: [-0.44, 0.15, 0.08], target: [0.0, 0.128, 0.0] },    // 3: Rear Battery / BMS / USB-C Service Bay
  { pos: [0.16, 0.232, 0.14], target: [0.008, 0.214, 0.0] },  // 4: Macro — 2-DOF Active Spatial Head & XIAO ESP32S3
  { pos: [0.20, 0.172, 0.20], target: [0.0, 0.150, 0.0] },    // 5: Macro — Torso PCA9685 / Dual Bucks / CoM IMU
  { pos: [0.18, 0.058, 0.18], target: [0.005, 0.028, 0.0] },  // 6: Macro — Coaxial 2-DOF Universal Ankle & Dual-Shear Clevis
  { pos: [0.64, 0.34, 0.64],  target: [0.0, 0.125, 0.0] },    // 7: Wide Studio Framing for Full 2-Stage Exploded View
];
let currentViewIdx = 0;
const targetCamPos = new THREE.Vector3(...VIEW_PRESETS[0].pos);
const targetLookAt = new THREE.Vector3(...VIEW_PRESETS[0].target);
let camLerping = false;

function applyCameraPreset(idx, instant = false) {
  const vp = VIEW_PRESETS[idx % VIEW_PRESETS.length];
  if (instant) {
    camera.position.set(...vp.pos);
    controls.target.set(...vp.target);
    controls.update();
  } else {
    targetCamPos.set(...vp.pos);
    targetLookAt.set(...vp.target);
    camLerping = true;
  }
}
applyCameraPreset(0, true);
controls.addEventListener('start', () => { camLerping = false; });

// --- Dynamic Selective Cross-Section Cutaway Planes (Slices Shells & Servo Cases, Keeps Internals Whole!) ---
const activeSectionPlanes = [];
const PLANE_SAGITTAL = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0.002);   // Slices right half (Z_three > +2mm)
const PLANE_CORONAL  = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 0.003);   // Slices front half (X_three > +3mm)
const PLANE_AXIAL    = new THREE.Plane(new THREE.Vector3(0, -1, 0), 0.168);   // Horizontal top-down torso floorplan cut
let cutawayMode = 0; // 0: Off, 1: Sagittal, 2: Coronal, 3: Axial

const mats = createAnatomyMaterials(activeSectionPlanes);
const edgeLineMat = new THREE.LineBasicMaterial({
  color: 0x1c2430, transparent: true, opacity: 0.26, clippingPlanes: activeSectionPlanes,
});
const witnessLineMat = new THREE.LineDashedMaterial({
  color: 0x64748b, dashSize: 0.0035, gapSize: 0.0025, transparent: true, opacity: 0.62,
});

const shellMeshes = [];
const edgeLines = [];
const explodeEntries = []; // { obj, basePos, explodeDir, stage, witnessLine, witnessAttr }
const harnessGroups = [];

function setOrthonormalBasis(obj, ex, ey, ez) {
  const m = new THREE.Matrix4().makeBasis(
    new THREE.Vector3(...ex),
    new THREE.Vector3(...ey),
    new THREE.Vector3(...ez)
  );
  obj.quaternion.setFromRotationMatrix(m);
}

function registerExplode(obj, dx, dy, dz, { stage = 1, witness = true } = {}) {
  const basePos = obj.position.clone();
  const explodeDir = new THREE.Vector3(dx * MM, dy * MM, dz * MM);
  let witnessLine = null;
  let witnessAttr = null;

  if (witness && obj.parent && explodeDir.length() > 0.004) {
    const pts = new Float32Array([
      basePos.x, basePos.y, basePos.z,
      basePos.x, basePos.y, basePos.z,
    ]);
    const g = new THREE.BufferGeometry();
    witnessAttr = new THREE.BufferAttribute(pts, 3);
    g.setAttribute('position', witnessAttr);
    witnessLine = new THREE.Line(g, witnessLineMat);
    witnessLine.visible = false;
    obj.parent.add(witnessLine);
  }

  explodeEntries.push({ obj, basePos, explodeDir, stage, witnessLine, witnessAttr });
}

// ============================================================================
// ARTICULATED 12-DOF HUMANOID SCENE GRAPH (Exact 0.01mm Coaxial Match to CAD!)
// ============================================================================
const worldRobot = new THREE.Group();
worldRobot.rotation.x = -Math.PI / 2;
scene.add(worldRobot);

const torsoGroup = new THREE.Group();
torsoGroup.position.set(0, 0, 124.0 * MM); // Global Z_HIP_ROLL = 124.0 mm
worldRobot.add(torsoGroup);

// --- 1. Torso Power, Control, IMU, Audio & Rear Service Hardware ---
const { packGroup, cells, bmsGroup, rearIOGroup } = createBattery2SAndRearIO(mats);
packGroup.position.set(-10.5 * MM, 0, 32.0 * MM);
torsoGroup.add(packGroup);
torsoGroup.add(rearIOGroup);
registerExplode(packGroup, -34, 0, 4, { stage: 1 });
registerExplode(cells[0], -10, -8, 12, { stage: 2 });
registerExplode(cells[1], -10,  8, 12, { stage: 2 });
registerExplode(bmsGroup, -14,  0,  0, { stage: 2 });
registerExplode(rearIOGroup, -44, 0, -4, { stage: 1 });

// 4x M2x6 Stainless Socket-Head Cap Screws on torso_back_cover (Y = +-26mm, Z = 10mm & 48mm)
for (const sy of [-26.0, 26.0]) {
  for (const dz of [10.0, 48.0]) {
    const bolt = createM2SocketBolt(mats, 6.5);
    setOrthonormalBasis(bolt, [0, 1, 0], [0, 0, 1], [-1, 0, 0]); // Head faces -X, shaft threads +X into torso
    bolt.position.set(-26.8 * MM, sy * MM, dz * MM);
    torsoGroup.add(bolt);
    registerExplode(bolt, -76, sy * 0.12, 0, { stage: 2 });
  }
}

const pca9685Group = createPCA9685Assembly(mats);
pca9685Group.position.set(1.2 * MM, 0, 33.0 * MM);
torsoGroup.add(pca9685Group);
registerExplode(pca9685Group, -14, 0, 16, { stage: 1 });

const imuCoMGroup = createMPU6050CoMAssembly(mats);
imuCoMGroup.position.set(2.0 * MM, 0, 14.0 * MM); // Exact Pelvis Cross-Tie Beam CoM Mount!
torsoGroup.add(imuCoMGroup);
registerExplode(imuCoMGroup, 14, 0, -16, { stage: 2 });

const { buck5A, cap1000, buck2A } = createDualBuckPowerNodes(mats);
buck5A.position.set(9.5 * MM, -21.0 * MM, 25.0 * MM);
torsoGroup.add(buck5A);
registerExplode(buck5A, 18, -30, 0, { stage: 1 });

cap1000.position.set(13.5 * MM, -14.0 * MM, 17.5 * MM);
torsoGroup.add(cap1000);
registerExplode(cap1000, 28, -22, -10, { stage: 2 });

buck2A.position.set(10.0 * MM, 21.0 * MM, 22.5 * MM);
torsoGroup.add(buck2A);
registerExplode(buck2A, 18, 30, -4, { stage: 1 });

const { ampGroup, spkGroup } = createAudioSystemNodes(mats);
ampGroup.position.set(11.5 * MM, 19.5 * MM, 39.5 * MM);
torsoGroup.add(ampGroup);
registerExplode(ampGroup, 24, 26, 10, { stage: 2 });

spkGroup.position.set(20.5 * MM, 0, 34.0 * MM); // Exact chest acoustic cavity!
torsoGroup.add(spkGroup);
registerExplode(spkGroup, 36, 0, 4, { stage: 1 });

// --- 2. Torso-Mounted Servos (Hip Roll L/R, Shoulder L/R, Neck Pan SG90) ---
for (const side of [1, -1]) {
  const sy = side * 23.0;
  // Hip Roll MG90S: spline along +X at (10.8, sy, 0), body extending +Z inside pelvis bulkhead!
  const svHipR = createPrecisionServo(mats, false);
  setOrthonormalBasis(svHipR, [0, 0, -1], [0, 1, 0], [1, 0, 0]);
  svHipR.position.set(10.8 * MM, sy * MM, 0);
  torsoGroup.add(svHipR);
  registerExplode(svHipR, 14, side * 10, 6, { stage: 2 });

  // Opposing Rear Pelvis Dual-Shear Bushing at (-12.5, sy, 0)
  const bushHipR = createDualShearBushing(mats);
  setOrthonormalBasis(bushHipR, [0, 1, 0], [0, 0, 1], [-1, 0, 0]);
  bushHipR.position.set(-12.5 * MM, sy * MM, 0);
  torsoGroup.add(bushHipR);
  registerExplode(bushHipR, -18, side * 10, 0, { stage: 2 });

  // Shoulder MG90S: spline along side*Y at (0, side*43.2, 46.0), body along -X inside ribcage!
  const svSh = createPrecisionServo(mats, false);
  setOrthonormalBasis(svSh, [1, 0, 0], [0, 0, -side], [0, side, 0]);
  svSh.position.set(0, side * 43.2 * MM, 46.0 * MM);
  torsoGroup.add(svSh);
  registerExplode(svSh, 0, side * 16, 8, { stage: 2 });
}

const neckPanServo = createPrecisionServo(mats, true); // Translucent SG90 CH10
neckPanServo.position.set(0, 0, 62.2 * MM);
torsoGroup.add(neckPanServo);
registerExplode(neckPanServo, 0, 0, 10, { stage: 2 });

// --- 3. 2-DOF Active Spatial Head & Sensor Pod ---
const headYawJoint = new THREE.Group();
headYawJoint.position.set(0, 0, 64.0 * MM); // Global Z = 188.0 mm
torsoGroup.add(headYawJoint);
registerExplode(headYawJoint, 0, 0, 22, { stage: 1 });

// Rotating Neck Yaw SG90 Servo Horn mounted on underside of neck_gimbal
const { group: yawHorn } = createServoHornAssembly(mats, { isSG90: true, armLenMM: 10.5 });
setOrthonormalBasis(yawHorn, [1, 0, 0], [0, -1, 0], [0, 0, -1]);
yawHorn.position.set(0, 0, -0.5 * MM);
headYawJoint.add(yawHorn);
registerExplode(yawHorn, 0, 0, -8, { stage: 2 });

// Head Tilt SG90 (CH11): spline along +Y at (0, +10.8, 16.0), body extending -Z inside neck_gimbal!
const headTiltServo = createPrecisionServo(mats, true);
setOrthonormalBasis(headTiltServo, [0, 0, 1], [1, 0, 0], [0, 1, 0]);
headTiltServo.position.set(0, 10.8 * MM, 16.0 * MM);
headYawJoint.add(headTiltServo);
registerExplode(headTiltServo, 0, 14, 0, { stage: 2 });

const headPitchJoint = new THREE.Group();
headPitchJoint.position.set(0, 0, 16.0 * MM); // Global Z = 204.0 mm
headYawJoint.add(headPitchJoint);
registerExplode(headPitchJoint, 0, 0, 28, { stage: 1 });

// Rotating Head Tilt SG90 Horn (+Y ear) & Opposing Dual-Shear Bushing (-Y ear)
const { group: tiltHorn } = createServoHornAssembly(mats, { isSG90: true, armLenMM: 9.5 });
setOrthonormalBasis(tiltHorn, [0, 0, 1], [1, 0, 0], [0, 1, 0]);
tiltHorn.position.set(0, 13.4 * MM, 0);
headPitchJoint.add(tiltHorn);
registerExplode(tiltHorn, 0, 16, 0, { stage: 2 });

const headTiltBush = createDualShearBushing(mats);
setOrthonormalBasis(headTiltBush, [1, 0, 0], [0, 0, 1], [0, -1, 0]);
headTiltBush.position.set(0, -13.4 * MM, 0);
headPitchJoint.add(headTiltBush);
registerExplode(headTiltBush, 0, -16, 0, { stage: 2 });

// Internal Head Compute & Coaxial Vision/Laser Suite (XIAO ESP32S3 Sense + VL53L0X)
const xiaoGroup = createXiaoESP32S3SenseAssembly(mats);
xiaoGroup.position.set(9.8 * MM, 0, 15.0 * MM); // Lens apex sits coaxially in head_visor Ø8.6mm port!
headPitchJoint.add(xiaoGroup);
registerExplode(xiaoGroup, 22, 0, 10, { stage: 2 });

const tofGroup = createVL53L0XAssembly(mats);
tofGroup.position.set(18.6 * MM, 0, 5.0 * MM); // Optical apertures coaxial with head_visor Z=+5.0mm window!
headPitchJoint.add(tofGroup);
registerExplode(tofGroup, 30, 0, -6, { stage: 2 });

// Active Spatial ToF 940nm Laser Cone & OV5640 62-deg Camera FOV Frustum
const sensorVizGroup = new THREE.Group();
headPitchJoint.add(sensorVizGroup);

const camConeGeo = new THREE.ConeGeometry(0.088, 0.16, 4, 1, true);
camConeGeo.rotateZ(Math.PI / 2);
camConeGeo.rotateX(Math.PI / 4);
const camConeWire = new THREE.LineSegments(
  new THREE.EdgesGeometry(camConeGeo),
  new THREE.LineBasicMaterial({ color: 0x0ea5e9, transparent: true, opacity: 0.28 })
);
camConeWire.position.set(0.102, 0, 15.0 * MM);
sensorVizGroup.add(camConeWire);

const tofBeamGeo = new THREE.CylinderGeometry(0.022, 0.0014, 0.17, 16, 1, true);
tofBeamGeo.rotateZ(-Math.PI / 2);
const tofBeam = new THREE.Mesh(
  tofBeamGeo,
  new THREE.MeshBasicMaterial({
    color: 0x06b6d4, transparent: true, opacity: 0.18, side: THREE.DoubleSide, depthWrite: false
  })
);
tofBeam.position.set(0.107, 0, 5.0 * MM);
sensorVizGroup.add(tofBeam);

const tofCenterRay = new THREE.Line(
  new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(21.0 * MM, 0, 5.0 * MM),
    new THREE.Vector3(192.0 * MM, 0, 5.0 * MM),
  ]),
  new THREE.LineBasicMaterial({ color: 0x0284c7, transparent: true, opacity: 0.58 })
);
sensorVizGroup.add(tofCenterRay);

// --- 4. Left & Right Arms + Rotating Shoulder Horns ---
function buildArmAssembly(side) {
  const shJoint = new THREE.Group();
  shJoint.position.set(0, side * 46.0 * MM, 46.0 * MM);
  torsoGroup.add(shJoint);
  registerExplode(shJoint, 0, side * 34, 0, { stage: 1 });

  const { group: shHorn } = createServoHornAssembly(mats, { isSG90: false, armLenMM: 11.5 });
  setOrthonormalBasis(shHorn, [0, 0, -1], [side, 0, 0], [0, side, 0]);
  shHorn.position.set(0, -side * 3.2 * MM, 0);
  shJoint.add(shHorn);
  registerExplode(shHorn, 0, -side * 10, 0, { stage: 2 });

  return shJoint;
}
const lShoulderJoint = buildArmAssembly(+1);
const rShoulderJoint = buildArmAssembly(-1);

// --- 5. Coaxial 2-DOF Universal-Ankle Legs (Exact 0.01mm Spline/Horn/Bushing Alignment!) ---
function buildLegChain(side) {
  const hipRoll = new THREE.Group();
  hipRoll.position.set(0, side * 23.0 * MM, 0);
  torsoGroup.add(hipRoll);
  registerExplode(hipRoll, 0, side * 18, -14, { stage: 1 });

  // Rotating Hip Roll Horn on front clevis ear (+X = +12.6mm)
  const { group: hrHorn } = createServoHornAssembly(mats, { armLenMM: 10.0 });
  setOrthonormalBasis(hrHorn, [0, 0, -1], [0, 1, 0], [1, 0, 0]);
  hrHorn.position.set(12.6 * MM, 0, 0);
  hipRoll.add(hrHorn);
  registerExplode(hrHorn, 14, 0, 0, { stage: 2 });

  // Hip Pitch MG90S inside hip_clevis (spline along +Y at Z = -18mm)
  const svHipP = createPrecisionServo(mats, false);
  setOrthonormalBasis(svHipP, [0, 0, -1], [-1, 0, 0], [0, 1, 0]);
  svHipP.position.set(0, (side * 3.0 + 10.4) * MM, -18.0 * MM);
  hipRoll.add(svHipP);
  registerExplode(svHipP, 0, 14, 6, { stage: 2 });

  const hipPitch = new THREE.Group();
  hipPitch.position.set(0, side * 3.0 * MM, -18.0 * MM);
  hipRoll.add(hipPitch);
  registerExplode(hipPitch, 0, side * 10, -18, { stage: 1 });

  // Rotating Hip Pitch Horn (+Y fork) & Opposing Dual-Shear Bushing (-Y fork)
  const { group: hpHorn } = createServoHornAssembly(mats, { armLenMM: 11.0 });
  setOrthonormalBasis(hpHorn, [0, 0, -1], [-1, 0, 0], [0, 1, 0]);
  hpHorn.position.set(0, 12.5 * MM, 0);
  hipPitch.add(hpHorn);
  registerExplode(hpHorn, 0, 14, 0, { stage: 2 });

  const hpBush = createDualShearBushing(mats);
  setOrthonormalBasis(hpBush, [1, 0, 0], [0, 0, 1], [0, -1, 0]);
  hpBush.position.set(0, -12.5 * MM, 0);
  hipPitch.add(hpBush);
  registerExplode(hpBush, 0, -14, 0, { stage: 2 });

  // Ankle Pitch MG90S at bottom of 88mm leg_link (spline along +Y at Z = -88mm)
  const svAnkP = createPrecisionServo(mats, false);
  setOrthonormalBasis(svAnkP, [0, 0, -1], [-1, 0, 0], [0, 1, 0]);
  svAnkP.position.set(0, 10.4 * MM, -88.0 * MM);
  hipPitch.add(svAnkP);
  registerExplode(svAnkP, 0, 14, 6, { stage: 2 });

  const anklePitch = new THREE.Group();
  anklePitch.position.set(0, 0, -88.0 * MM);
  hipPitch.add(anklePitch);
  registerExplode(anklePitch, 0, 0, -20, { stage: 1 });

  // Rotating Ankle Pitch Horn (+Y ear) & Opposing Dual-Shear Bushing (-Y ear)
  const { group: apHorn } = createServoHornAssembly(mats, { armLenMM: 9.5 });
  setOrthonormalBasis(apHorn, [0, 0, 1], [1, 0, 0], [0, 1, 0]);
  apHorn.position.set(0, 12.5 * MM, 0);
  anklePitch.add(apHorn);
  registerExplode(apHorn, 0, 14, 0, { stage: 2 });

  const apBush = createDualShearBushing(mats);
  setOrthonormalBasis(apBush, [1, 0, 0], [0, 0, 1], [0, -1, 0]);
  apBush.position.set(0, -12.5 * MM, 0);
  anklePitch.add(apBush);
  registerExplode(apBush, 0, -14, 0, { stage: 2 });

  // Ankle Roll MG90S coaxially inside ankle_clevis (spline along +X at Z = 0)
  const svAnkR = createPrecisionServo(mats, false);
  setOrthonormalBasis(svAnkR, [0, 0, -1], [0, 1, 0], [1, 0, 0]);
  svAnkR.position.set(10.6 * MM, 0, 0);
  anklePitch.add(svAnkR);
  registerExplode(svAnkR, 14, 0, 6, { stage: 2 });

  const ankleRoll = new THREE.Group();
  ankleRoll.position.set(0, 0, 0);
  anklePitch.add(ankleRoll);
  registerExplode(ankleRoll, 0, 0, -18, { stage: 1 });

  // Rotating Ankle Roll Horn on front foot tower (+X = +13.0mm) & Rear Opposing Bushing (-X = -12.6mm)
  const { group: arHorn } = createServoHornAssembly(mats, { armLenMM: 10.5 });
  setOrthonormalBasis(arHorn, [0, 1, 0], [0, 0, 1], [1, 0, 0]);
  arHorn.position.set(13.0 * MM, 0, 0);
  ankleRoll.add(arHorn);
  registerExplode(arHorn, 15, 0, 0, { stage: 2 });

  const arBush = createDualShearBushing(mats);
  setOrthonormalBasis(arBush, [0, 1, 0], [0, 0, 1], [-1, 0, 0]);
  arBush.position.set(-12.6 * MM, 0, 0);
  ankleRoll.add(arBush);
  registerExplode(arBush, -15, 0, 0, { stage: 2 });

  // High-Friction 2.0 mm Silicone Sole Pad recessed into foot_sole bottom pocket (mu >= 0.90)
  const pad = new THREE.Mesh(new THREE.BoxGeometry(43.6 * MM, 39.6 * MM, 2.0 * MM), mats.soleSilicone);
  pad.position.set(3.0 * MM, 0, -17.5 * MM);
  pad.receiveShadow = true;
  ankleRoll.add(pad);
  registerExplode(pad, 0, 0, -16, { stage: 2 });

  return { hipRoll, hipPitch, anklePitch, ankleRoll };
}

const leftLeg  = buildLegChain(+1);
const rightLeg = buildLegChain(-1);

// --- 6. Complete Color-Coded 3D Silicone Power, I2C, I2S & Internal Conduit Wiring Harnesses ---
const harnesses = createCompleteWiringHarnesses(mats);
torsoGroup.add(harnesses.torsoHarness);
headPitchJoint.add(harnesses.headHarness);
leftLeg.hipPitch.add(harnesses.leftLegHarness);
rightLeg.hipPitch.add(harnesses.rightLegHarness);
harnessGroups.push(
  harnesses.torsoHarness,
  harnesses.headHarness,
  harnesses.leftLegHarness,
  harnesses.rightLegHarness
);

// --- 7. Kinematic 12-DOF Joint Axis Vectors + Whole-Body CoM Plumb-Line Indicator ---
const kinematicMarkers = attachKinematicAndCoMGuides({
  torsoGroup, headYawJoint, headPitchJoint, lShoulderJoint, rShoulderJoint, leftLeg, rightLeg
});

// ============================================================================
// DEDUPLICATED & CREASE-SMOOTHED OPENSCAD WATERTIGHT STL LOADER
// ============================================================================
const stlLoader = new STLLoader();
const stlCache = new Map();
let showEdges = false;

function loadCachedSTL(url) {
  if (!stlCache.has(url)) {
    stlCache.set(url, new Promise((resolve, reject) => {
      stlLoader.load(url, (rawGeo) => {
        rawGeo.scale(MM, MM, MM);
        computeCreasedNormals(rawGeo, 30.0);
        const edgeGeo = new THREE.EdgesGeometry(rawGeo, 26);
        resolve({ geo: rawGeo, edgeGeo });
      }, undefined, reject);
    }));
  }
  return stlCache.get(url);
}

function mountSTL(url, parent, mat, offsetMM = [0, 0, 0], explodeMM = null, explodeStage = 1) {
  loadCachedSTL(url).then(({ geo, edgeGeo }) => {
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(offsetMM[0] * MM, offsetMM[1] * MM, offsetMM[2] * MM);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    parent.add(mesh);
    shellMeshes.push(mesh);

    // Apply current anatomy shell visibility if loaded after user clicked X-Ray
    mesh.visible = (xrayStage < 2);

    const edges = new THREE.LineSegments(edgeGeo, edgeLineMat);
    edges.visible = showEdges;
    mesh.add(edges);
    edgeLines.push(edges);

    if (explodeMM) registerExplode(mesh, ...explodeMM, { stage: explodeStage });
  });
}

mountSTL('cad/stl/torso_chassis.stl',     torsoGroup,         mats.shellPrimary, [0, 0, 0],         null);
mountSTL('cad/stl/torso_back_cover.stl',  torsoGroup,         mats.shellAccent,  [0, 0, 0],         [-58, 0, 0], 1);
mountSTL('cad/stl/neck_gimbal.stl',       headYawJoint,       mats.shellAccent,  [0, 0, 0],         null);
mountSTL('cad/stl/head_shell.stl',        headPitchJoint,     mats.shellPrimary, [0, 0, 0],         null);
mountSTL('cad/stl/head_visor.stl',        headPitchJoint,     mats.visor,        [0, 0, 0],         [42, 0, 0],  1);
mountSTL('cad/stl/arm_link_left.stl',     lShoulderJoint,     mats.shellPrimary, [0, 0, 0],         null);
mountSTL('cad/stl/arm_link_right.stl',    rShoulderJoint,     mats.shellPrimary, [0, 0, 0],         null);

// Left & Right Leg STLs (cached — zero duplicate HTTP fetches!)
mountSTL('cad/stl/hip_clevis_left.stl',   leftLeg.hipRoll,    mats.shellAccent,  [0, 3.0, -18.0],   null);
mountSTL('cad/stl/leg_link.stl',          leftLeg.hipPitch,   mats.shellPrimary, [0, 0, -88.0],     null);
mountSTL('cad/stl/ankle_clevis.stl',      leftLeg.anklePitch, mats.shellAccent,  [0, 0, 0],         null);
mountSTL('cad/stl/foot_sole.stl',         leftLeg.ankleRoll,  mats.shellPrimary, [0, 0, -18.0],     null);

mountSTL('cad/stl/hip_clevis_right.stl',  rightLeg.hipRoll,   mats.shellAccent,  [0, -3.0, -18.0],  null);
mountSTL('cad/stl/leg_link.stl',          rightLeg.hipPitch,  mats.shellPrimary, [0, 0, -88.0],     null);
mountSTL('cad/stl/ankle_clevis.stl',      rightLeg.anklePitch,mats.shellAccent,  [0, 0, 0],         null);
mountSTL('cad/stl/foot_sole.stl',         rightLeg.ankleRoll, mats.shellPrimary, [0, 0, -18.0],     null);

// ============================================================================
// PURE ICON-ONLY VIEWPORT CONTROLS & MULTI-STAGE ANATOMY MODES
// ============================================================================
let isPlaying = true;
let xrayStage = 0; // 0: Solid Shell, 1: Ghost X-Ray + Transparent Servo Gearboxes, 2: Naked Mechatronics Skeleton
let showSensors = true;
let explodeFactor = 0.0;
let simTime = 0.0;

const btnPlay      = document.getElementById('btn-play');
const btnXray      = document.getElementById('btn-xray');
const btnCutaway   = document.getElementById('btn-cutaway');
const btnEdges     = document.getElementById('btn-edges');
const btnSensors   = document.getElementById('btn-sensors');
const btnView      = document.getElementById('btn-view');
const btnReset     = document.getElementById('btn-reset');
const btnFull      = document.getElementById('btn-fullscreen');
const btnInfo      = document.getElementById('btn-info');
const btnCloseInfo = document.getElementById('btn-close-drawer');
const btnExplode   = document.getElementById('btn-explode-toggle');
const sliderExp    = document.getElementById('slider-explode');
const specDrawer   = document.getElementById('spec-drawer');

function applyXrayStage(stage) {
  xrayStage = ((stage % 3) + 3) % 3;
  btnXray.classList.toggle('is-active', xrayStage > 0);
  btnXray.dataset.stage = String(xrayStage);

  const shellsVisible = (xrayStage < 2);
  for (const m of shellMeshes) m.visible = shellsVisible;

  const isGhost = (xrayStage === 1);
  for (const sm of [mats.shellPrimary, mats.shellAccent]) {
    sm.transparent = isGhost;
    sm.opacity = isGhost ? 0.16 : 1.0;
    sm.depthWrite = !isGhost;
    sm.needsUpdate = true;
  }
  mats.visor.transparent = isGhost;
  mats.visor.opacity = isGhost ? 0.22 : 1.0;
  mats.visor.depthWrite = !isGhost;
  mats.visor.needsUpdate = true;

  // Reveal internal 4-stage metal gear train & coreless DC motor inside MG90S servos in X-Ray / Skeleton modes!
  const servoTransparent = (xrayStage >= 1);
  mats.servoMG90S.transparent = servoTransparent;
  mats.servoMG90S.opacity = (xrayStage === 1) ? 0.24 : (xrayStage === 2 ? 0.34 : 1.0);
  mats.servoMG90S.depthWrite = !servoTransparent;
  mats.servoMG90S.needsUpdate = true;

  mats.servoSG90.opacity = servoTransparent ? 0.26 : 0.72;
  mats.servoSG90.depthWrite = !servoTransparent;
  mats.servoSG90.needsUpdate = true;
}

function applyCutawayMode(mode) {
  cutawayMode = ((mode % 4) + 4) % 4;
  activeSectionPlanes.length = 0;
  if (cutawayMode === 1) activeSectionPlanes.push(PLANE_SAGITTAL);
  else if (cutawayMode === 2) activeSectionPlanes.push(PLANE_CORONAL);
  else if (cutawayMode === 3) activeSectionPlanes.push(PLANE_AXIAL);

  if (btnCutaway) {
    btnCutaway.classList.toggle('is-active', cutawayMode > 0);
    btnCutaway.dataset.mode = String(cutawayMode);
  }
}

function applyEdgesMode(enabled) {
  showEdges = enabled;
  btnEdges.classList.toggle('is-active', showEdges);
  for (const l of edgeLines) l.visible = showEdges;
  for (const km of kinematicMarkers) km.visible = showEdges;
}

function resetPoseToNeutral() {
  torsoGroup.position.set(0, 0, 124.0 * MM);
  torsoGroup.quaternion.identity();
  leftLeg.hipRoll.rotation.x = 0; leftLeg.hipPitch.rotation.y = 0;
  leftLeg.anklePitch.rotation.y = 0; leftLeg.ankleRoll.rotation.x = 0;
  rightLeg.hipRoll.rotation.x = 0; rightLeg.hipPitch.rotation.y = 0;
  rightLeg.anklePitch.rotation.y = 0; rightLeg.ankleRoll.rotation.x = 0;
  lShoulderJoint.rotation.y = 0; rShoulderJoint.rotation.y = 0;
  headYawJoint.rotation.z = 0; headPitchJoint.rotation.y = 0;
}

function smoothStep01(x) {
  const t = Math.max(0, Math.min(1, x));
  return t * t * (3.0 - 2.0 * t);
}

function setExplode(val01) {
  explodeFactor = Math.max(0, Math.min(1, val01));
  sliderExp.value = String(Math.round(explodeFactor * 100));
  btnExplode.classList.toggle('is-active', explodeFactor > 0.02);

  if (explodeFactor > 0.02 && isPlaying) {
    isPlaying = false;
    btnPlay.classList.remove('is-active');
    resetPoseToNeutral();
  }

  // Stage 1 (0..0.62): Sub-assemblies & service covers separate
  // Stage 2 (0.35..1.0): Individual servos, horns, M2 bolts, bushings, cells & pads extract coaxially!
  const tStage1 = smoothStep01(explodeFactor / 0.62);
  const tStage2 = smoothStep01((explodeFactor - 0.35) / 0.65);

  for (const item of explodeEntries) {
    const f = item.stage === 2 ? tStage2 : tStage1;
    item.obj.position.copy(item.basePos).addScaledVector(item.explodeDir, f);
    if (item.witnessLine && item.witnessAttr) {
      const showW = f > 0.03;
      item.witnessLine.visible = showW;
      if (showW) {
        item.witnessAttr.setXYZ(1, item.obj.position.x, item.obj.position.y, item.obj.position.z);
        item.witnessAttr.needsUpdate = true;
        item.witnessLine.computeLineDistances();
      }
    }
  }

  // Hide flexible wire harnesses cleanly during wide explosion so mechanical coaxial lines stay crystal clear
  const showHarness = explodeFactor < 0.22;
  for (const hg of harnessGroups) hg.visible = showHarness;
}

btnPlay.addEventListener('click', () => {
  isPlaying = !isPlaying;
  btnPlay.classList.toggle('is-active', isPlaying);
  if (isPlaying && explodeFactor > 0.0) setExplode(0.0);
});

btnXray.addEventListener('click', () => applyXrayStage(xrayStage + 1));
if (btnCutaway) {
  btnCutaway.addEventListener('click', () => applyCutawayMode(cutawayMode + 1));
}
btnEdges.addEventListener('click', () => applyEdgesMode(!showEdges));

btnSensors.addEventListener('click', () => {
  showSensors = !showSensors;
  btnSensors.classList.toggle('is-active', showSensors);
  sensorVizGroup.visible = showSensors;
});

btnView.addEventListener('click', () => {
  currentViewIdx = (currentViewIdx + 1) % VIEW_PRESETS.length;
  applyCameraPreset(currentViewIdx, false);
});

btnReset.addEventListener('click', () => {
  currentViewIdx = 0;
  applyCameraPreset(0, false);
  setExplode(0.0);
  applyXrayStage(0);
  applyCutawayMode(0);
  applyEdgesMode(false);
});

btnExplode.addEventListener('click', () => {
  const nextExp = explodeFactor > 0.1 ? 0.0 : 1.0;
  setExplode(nextExp);
  applyCameraPreset(nextExp > 0.5 ? 7 : 0, false);
});

sliderExp.addEventListener('input', (e) => {
  setExplode(Number(e.target.value) / 100.0);
});

btnFull.addEventListener('click', async () => {
  try {
    if (!document.fullscreenElement) {
      await document.documentElement.requestFullscreen();
      if (screen.orientation && screen.orientation.lock) {
        try { await screen.orientation.lock('landscape'); } catch (_) {}
      }
      btnFull.classList.add('is-active');
    } else {
      await document.exitFullscreen();
      btnFull.classList.remove('is-active');
    }
  } catch (_) {}
});

function toggleDrawer(open) {
  specDrawer.classList.toggle('is-open', open);
  specDrawer.setAttribute('aria-hidden', open ? 'false' : 'true');
  btnInfo.classList.toggle('is-active', open);
}
btnInfo.addEventListener('click', () => toggleDrawer(!specDrawer.classList.contains('is-open')));
btnCloseInfo.addEventListener('click', () => toggleDrawer(false));

// Interactive Subsystem Anatomy Focus Preset Trigger (callable from inside #spec-drawer)
const ANATOMY_PRESETS = {
  sagittal_cut: { view: 2, xray: 0, cutaway: 1, explode: 0.0,  edges: true  },
  xray_gears:   { view: 0, xray: 1, cutaway: 0, explode: 0.0,  edges: true  },
  naked_core:   { view: 0, xray: 2, cutaway: 0, explode: 0.0,  edges: false },
  head_ai:      { view: 4, xray: 1, cutaway: 1, explode: 0.0,  edges: false },
  torso_ctrl:   { view: 5, xray: 1, cutaway: 2, explode: 0.0,  edges: true  },
  power_2s:     { view: 3, xray: 1, cutaway: 0, explode: 0.35, edges: false },
  ankle_mech:   { view: 6, xray: 1, cutaway: 0, explode: 0.55, edges: true  },
  full_explode: { view: 7, xray: 0, cutaway: 0, explode: 1.0,  edges: false },
};

document.querySelectorAll('[data-anatomy-preset]').forEach((btn) => {
  btn.addEventListener('click', () => {
    const key = btn.getAttribute('data-anatomy-preset');
    const cfg = ANATOMY_PRESETS[key];
    if (!cfg) return;
    currentViewIdx = cfg.view;
    applyCameraPreset(cfg.view, false);
    applyXrayStage(cfg.xray);
    applyCutawayMode(cfg.cutaway);
    applyEdgesMode(cfg.edges);
    setExplode(cfg.explode);
    if (window.innerWidth < 760) toggleDrawer(false);
  });
});

// Populate Explanation Drawer Tables Dynamically from Verified Data (window.RNV1_DATA)
function populateExplanationDrawer() {
  const data = window.RNV1_DATA;
  if (!data) return;

  const policyEl = document.getElementById('policy-summary-container');
  if (policyEl && data.policy) {
    const p = data.policy;
    const rs = (data.replay && data.replay.summary) ? data.replay.summary : null;
    const replayMetric = rs
      ? `${rs.dist_x_m.toFixed(3)} m / 10s（最大傾き ${rs.max_tilt_deg.toFixed(1)}°・RMSトルク ${rs.rms_torque_nm.toFixed(4)} N·m・${rs.fell ? '転倒あり' : '無転倒'}）`
      : 'MuJoCo 実物理リプレイ同期済';
    policyEl.innerHTML = `
      <table class="spec-table">
        <tbody>
          <tr><th>学習・最適化</th><td>${p.training_origin}</td></tr>
          <tr><th>ネットワーク構成</th><td><code>${p.architecture}</code>（${p.control_hz} Hz・残差スケール ${p.residual_scale}）</td></tr>
          <tr><th>パラメータ / Flash</th><td>${p.total_params.toLocaleString()} params（${(p.flash_bytes / 1024).toFixed(1)} KiB [${p.flash_bytes.toLocaleString()} B]・4-way SIMD展開 ${p.macs_per_step.toLocaleString()} MACs）</td></tr>
          <tr><th>組込CPG周期</th><td>${p.cpg_freq_hz} Hz（roll ${p.cpg_roll_amp_rad} rad / stride ${p.cpg_stride_amp_rad} rad）</td></tr>
          <tr><th>実物理歩行評価</th><td>${replayMetric}</td></tr>
        </tbody>
      </table>
    `;
  }

  const videoWrap = document.getElementById('policy-video-container');
  const videoEl = document.getElementById('rnv1-eval-video');
  if (videoWrap && videoEl && data.policy && data.policy.eval_video_path) {
    videoEl.src = data.policy.eval_video_path;
    videoWrap.hidden = false;
  }

  const bomEl = document.getElementById('bom-table-container');
  if (bomEl && data.bom) {
    let rows = data.bom.components.map((c) => `
      <tr>
        <td>${c.name_ja}</td>
        <td class="num">¥${c.price_optimal_jpy.toLocaleString()}</td>
        <td class="num">¥${c.price_domestic_jpy.toLocaleString()}</td>
        <td class="num">${c.total_mass_g.toFixed(1)} g</td>
      </tr>
    `).join('');
    rows += `
      <tr class="total-row">
        <td>合計（予算上限 ¥10,000）</td>
        <td class="num">¥${data.bom.summary.total_price_optimal_jpy.toLocaleString()}</td>
        <td class="num">¥${data.bom.summary.total_price_domestic_jpy.toLocaleString()}</td>
        <td class="num">${data.bom.summary.total_mass_g.toFixed(1)} g</td>
      </tr>
    `;
    bomEl.innerHTML = `
      <table class="spec-table">
        <thead><tr><th>部品構成</th><th class="num">最適調達</th><th class="num">国内通販</th><th class="num">質量</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  const simEl = document.getElementById('sim-results-container');
  if (simEl && data.verification) {
    const suites = data.verification.suites || [];
    const sRows = suites.map((s) => {
      const topMetrics = Object.entries(s.metrics || {}).slice(0, 3)
        .map(([k, v]) => `${k}: ${v}`).join(' / ');
      const tagClass = s.passed ? 'pass-tag' : 'fail-tag';
      return `
        <tr>
          <td><span class="${tagClass}">${s.passed ? '100% PASS' : 'FAIL'}</span></td>
          <td><strong>${s.suite}</strong><br><span>${topMetrics}</span></td>
        </tr>
      `;
    }).join('');
    simEl.innerHTML = `
      <p>全 <strong>${data.verification.passed_suites} / ${data.verification.total_suites}</strong> 検証スイート合格（MuJoCo 4.0.1 / PyBullet 3.2.7 / OpenSCAD+Trimesh / SciPy FEA・回路・熱 / g++ C++ビット完全一致）。</p>
      <table class="spec-table"><tbody>${sRows}</tbody></table>
    `;
  }
}
populateExplanationDrawer();

// --- Main 60 FPS Animation Loop Replaying Real Recorded MuJoCo RNV1 Physics Frames ---
const clock = new THREE.Clock();
const tmpQ1 = new THREE.Quaternion();
const tmpQ2 = new THREE.Quaternion();

function applyMuJoCoReplay(tSec) {
  const frames = window.RNV1_DATA && window.RNV1_DATA.replay && window.RNV1_DATA.replay.frames;
  if (!frames || frames.length < 2) return;
  const hz = window.RNV1_DATA.replay.control_hz || 50.0;
  const fFloat = (tSec * hz) % (frames.length - 1);
  const i0 = Math.floor(fFloat);
  const i1 = (i0 + 1) % frames.length;
  const u = fFloat - i0;
  const a = frames[i0], b = frames[i1];
  const lerp = (x, y) => x + (y - x) * u;

  torsoGroup.position.set(0, 0, lerp(a.pos[2], b.pos[2]));
  tmpQ1.set(a.quat[1], a.quat[2], a.quat[3], a.quat[0]);
  tmpQ2.set(b.quat[1], b.quat[2], b.quat[3], b.quat[0]);
  torsoGroup.quaternion.copy(tmpQ1).slerp(tmpQ2, u);

  const ja = a.joints, jb = b.joints;
  leftLeg.hipRoll.rotation.x     = lerp(ja.l_hip_roll, jb.l_hip_roll);
  leftLeg.hipPitch.rotation.y    = lerp(ja.l_hip_pitch, jb.l_hip_pitch);
  leftLeg.anklePitch.rotation.y  = lerp(ja.l_ankle_pitch, jb.l_ankle_pitch);
  leftLeg.ankleRoll.rotation.x   = lerp(ja.l_ankle_roll, jb.l_ankle_roll);

  rightLeg.hipRoll.rotation.x    = lerp(ja.r_hip_roll, jb.r_hip_roll);
  rightLeg.hipPitch.rotation.y   = lerp(ja.r_hip_pitch, jb.r_hip_pitch);
  rightLeg.anklePitch.rotation.y = lerp(ja.r_ankle_pitch, jb.r_ankle_pitch);
  rightLeg.ankleRoll.rotation.x  = lerp(ja.r_ankle_roll, jb.r_ankle_roll);

  lShoulderJoint.rotation.y      = lerp(ja.l_shoulder, jb.l_shoulder);
  rShoulderJoint.rotation.y      = lerp(ja.r_shoulder, jb.r_shoulder);
  headYawJoint.rotation.z        = lerp(ja.head_yaw, jb.head_yaw);
  headPitchJoint.rotation.y      = lerp(ja.head_pitch, jb.head_pitch);
}

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);

  if (isPlaying && explodeFactor < 0.01) {
    simTime += dt;
    applyMuJoCoReplay(simTime);
  }

  if (camLerping) {
    const k = 1.0 - Math.exp(-9.5 * dt);
    camera.position.lerp(targetCamPos, k);
    controls.target.lerp(targetLookAt, k);
    if (camera.position.distanceTo(targetCamPos) < 0.0015) camLerping = false;
  }
  controls.update();
  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
