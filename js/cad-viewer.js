import * as THREE from '../vendor/three/three.module.min.js';
import { OrbitControls } from '../vendor/three/OrbitControls.js';
import { STLLoader } from '../vendor/three/STLLoader.js';

const MM = 0.001; // Millimeters -> meters

// --- Setup Pure White (#FFFFFF) Studio WebGL Renderer ---
const canvas = document.getElementById('cad-canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0xffffff, 1);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.06;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);

// --- Studio Three-Point Industrial CAD Lighting ---
const hemiLight = new THREE.HemisphereLight(0xffffff, 0xe8edf5, 1.05);
hemiLight.position.set(0, 1.0, 0);
scene.add(hemiLight);

const keyLight = new THREE.DirectionalLight(0xffffff, 1.35);
keyLight.position.set(0.55, 0.95, 0.72);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
keyLight.shadow.camera.near = 0.1;
keyLight.shadow.camera.far = 3.0;
const sc = 0.35;
keyLight.shadow.camera.left = -sc;
keyLight.shadow.camera.right = sc;
keyLight.shadow.camera.top = sc;
keyLight.shadow.camera.bottom = -sc;
keyLight.shadow.bias = -0.0004;
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0xf2f6ff, 0.55);
fillLight.position.set(-0.65, 0.45, -0.40);
scene.add(fillLight);

const rimLight = new THREE.DirectionalLight(0xffffff, 0.48);
rimLight.position.set(-0.25, 0.75, 0.65);
scene.add(rimLight);

// Invisible Shadow-Catcher Plane on Pure White Studio Floor (Y = 0)
const shadowFloor = new THREE.Mesh(
  new THREE.PlaneGeometry(2.4, 2.4),
  new THREE.ShadowMaterial({ opacity: 0.13 })
);
shadowFloor.rotation.x = -Math.PI / 2;
shadowFloor.position.y = 0;
shadowFloor.receiveShadow = true;
scene.add(shadowFloor);

// --- Camera & Damped OrbitControls ---
const camera = new THREE.PerspectiveCamera(34, window.innerWidth / window.innerHeight, 0.01, 20);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.minDistance = 0.18;
controls.maxDistance = 1.25;
controls.maxPolarAngle = Math.PI * 0.495;

const VIEW_PRESETS = [
  { pos: [0.38, 0.23, 0.38], target: [0.0, 0.118, 0.0] },   // 3/4 Isometric Hero
  { pos: [0.48, 0.13, 0.00], target: [0.0, 0.118, 0.0] },   // Front Elevation
  { pos: [0.00, 0.13, 0.48], target: [0.0, 0.118, 0.0] },   // Right Profile
  { pos: [-0.48, 0.15, 0.05], target: [0.0, 0.118, 0.0] },  // Rear Service Bay
  { pos: [0.06, 0.54, 0.06], target: [0.0, 0.118, 0.0] },   // Top Orthographic-style
];
let currentViewIdx = 0;

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
const targetCamPos = new THREE.Vector3(...VIEW_PRESETS[0].pos);
const targetLookAt = new THREE.Vector3(...VIEW_PRESETS[0].target);
let camLerping = false;
applyCameraPreset(0, true);
controls.addEventListener('start', () => { camLerping = false; });

// --- Physically Based Industrial CAD Materials ---
const shellMaterials = [];
function makeShellMat(hex = 0xf4f6f9, rough = 0.42, metal = 0.05) {
  const m = new THREE.MeshStandardMaterial({
    color: hex, roughness: rough, metalness: metal,
    transparent: false, opacity: 1.0, side: THREE.DoubleSide
  });
  shellMaterials.push(m);
  return m;
}

const matShellPrimary   = makeShellMat(0xf4f6f9, 0.40, 0.05); // Matte Pearl-White PETG/PLA+
const matShellAccent    = makeShellMat(0xe5e9f0, 0.48, 0.08); // Light Titanium-Gray Structural Brackets
const matVisor          = makeShellMat(0x161b24, 0.14, 0.35); // Glossy Smoked Optical Visor
const matSoleSilicone   = new THREE.MeshStandardMaterial({ color: 0x26292e, roughness: 0.88, metalness: 0.02 });
const matServoMG90S     = new THREE.MeshStandardMaterial({ color: 0x1d4d8f, roughness: 0.36, metalness: 0.12 });
const matServoSG90      = new THREE.MeshStandardMaterial({ color: 0x2e7fd1, roughness: 0.32, metalness: 0.08, transparent: true, opacity: 0.90 });
const matBrass          = new THREE.MeshStandardMaterial({ color: 0xc8a452, roughness: 0.28, metalness: 0.85 });
const matAluminum       = new THREE.MeshStandardMaterial({ color: 0xcbd2db, roughness: 0.26, metalness: 0.82 });
const matHeatsink       = new THREE.MeshStandardMaterial({ color: 0x1e2229, roughness: 0.35, metalness: 0.72 });
const matPCBBlack       = new THREE.MeshStandardMaterial({ color: 0x191d24, roughness: 0.50, metalness: 0.10 });
const matPCBBlue        = new THREE.MeshStandardMaterial({ color: 0x14417a, roughness: 0.48, metalness: 0.10 });
const matPCBPurple      = new THREE.MeshStandardMaterial({ color: 0x562c7a, roughness: 0.48, metalness: 0.10 });
const matPCBGreen       = new THREE.MeshStandardMaterial({ color: 0x1a683f, roughness: 0.50, metalness: 0.10 });
const matBatteryCopper  = new THREE.MeshStandardMaterial({ color: 0xc26a2e, roughness: 0.34, metalness: 0.45 });
const matLensGlass      = new THREE.MeshPhysicalMaterial({ color: 0x0b1629, roughness: 0.06, metalness: 0.15, clearcoat: 1.0 });
const edgeLineMat       = new THREE.LineBasicMaterial({ color: 0x1c2430, transparent: true, opacity: 0.24 });

const edgeLines = [];
const explodeEntries = []; // { obj, basePos: Vector3, explodeDir: Vector3 }

function registerExplode(obj, dx, dy, dz) {
  explodeEntries.push({
    obj,
    basePos: obj.position.clone(),
    explodeDir: new THREE.Vector3(dx * MM, dy * MM, dz * MM),
  });
}

// --- Procedural Precision Electronic CAD Component Builders (in OpenSCAD mm -> m) ---
function createMG90S(isSG90 = false) {
  const g = new THREE.Group();
  const mat = isSG90 ? matServoSG90 : matServoMG90S;
  const body = new THREE.Mesh(new THREE.BoxGeometry(22.8*MM, 12.2*MM, 22.0*MM), mat);
  body.castShadow = true;
  g.add(body);
  const flange = new THREE.Mesh(new THREE.BoxGeometry(32.2*MM, 12.2*MM, 2.4*MM), mat);
  flange.position.z = 5.5*MM;
  g.add(flange);
  const dome = new THREE.Mesh(new THREE.CylinderGeometry(5.8*MM, 5.8*MM, 4.5*MM, 20), isSG90 ? mat : matAluminum);
  dome.rotation.x = Math.PI / 2;
  dome.position.set(5.5*MM, 0, 12.5*MM);
  g.add(dome);
  const spline = new THREE.Mesh(new THREE.CylinderGeometry(2.4*MM, 2.4*MM, 3.6*MM, 16), matBrass);
  spline.rotation.x = Math.PI / 2;
  spline.position.set(5.5*MM, 0, 16.0*MM);
  g.add(spline);
  return g;
}

function createXiaoESP32S3Sense() {
  const g = new THREE.Group();
  // Main XIAO PCB (21 x 17.8 x 1.6mm) facing +X
  const pcb = new THREE.Mesh(new THREE.BoxGeometry(2.0*MM, 17.8*MM, 21.0*MM), matPCBBlack);
  g.add(pcb);
  // Dual black anodized aluminum heatsinks on rear
  const hs = new THREE.Mesh(new THREE.BoxGeometry(3.2*MM, 11.0*MM, 11.0*MM), matHeatsink);
  hs.position.x = -2.6*MM;
  g.add(hs);
  // Sense camera expansion daughterboard + OV5640/OV3660 lens barrel facing +X
  const camPcb = new THREE.Mesh(new THREE.BoxGeometry(1.6*MM, 17.8*MM, 21.0*MM), matPCBBlack);
  camPcb.position.x = 5.5*MM;
  g.add(camPcb);
  const lensBarrel = new THREE.Mesh(new THREE.CylinderGeometry(4.0*MM, 4.2*MM, 5.0*MM, 24), matHeatsink);
  lensBarrel.rotation.z = Math.PI / 2;
  lensBarrel.position.set(8.6*MM, 0, 0);
  g.add(lensBarrel);
  const lensGlass = new THREE.Mesh(new THREE.CylinderGeometry(2.6*MM, 2.6*MM, 0.8*MM, 20), matLensGlass);
  lensGlass.rotation.z = Math.PI / 2;
  lensGlass.position.set(11.1*MM, 0, 0);
  g.add(lensGlass);
  // Gold castellated pads
  for (let side of [-1, 1]) {
    const pads = new THREE.Mesh(new THREE.BoxGeometry(1.4*MM, 1.4*MM, 18.0*MM), matBrass);
    pads.position.set(0, side * 8.4*MM, 0);
    g.add(pads);
  }
  return g;
}

function createVL53L0X() {
  const g = new THREE.Group();
  const pcb = new THREE.Mesh(new THREE.BoxGeometry(1.6*MM, 13.3*MM, 10.5*MM), matPCBPurple);
  g.add(pcb);
  const sensor = new THREE.Mesh(new THREE.BoxGeometry(1.4*MM, 4.4*MM, 2.4*MM), matLensGlass);
  sensor.position.x = 1.4*MM;
  g.add(sensor);
  return g;
}

function createBatteryPack2S() {
  const g = new THREE.Group();
  for (let sy of [-8.0, 8.0]) {
    const cell = new THREE.Mesh(new THREE.CylinderGeometry(7.0*MM, 7.0*MM, 50.0*MM, 24), matBatteryCopper);
    cell.rotation.x = Math.PI / 2; // Vertical along Z in OpenSCAD frame
    cell.position.set(0, sy*MM, 0);
    cell.castShadow = true;
    g.add(cell);
    const cap = new THREE.Mesh(new THREE.CylinderGeometry(5.5*MM, 5.5*MM, 1.4*MM, 16), matAluminum);
    cap.rotation.x = Math.PI / 2;
    cap.position.set(0, sy*MM, 25.5*MM);
    g.add(cap);
  }
  const bms = new THREE.Mesh(new THREE.BoxGeometry(3.0*MM, 28.0*MM, 10.0*MM), matPCBGreen);
  bms.position.set(-7.5*MM, 0, -18.0*MM);
  g.add(bms);
  return g;
}

function createSpeaker28mm() {
  const g = new THREE.Group();
  const frame = new THREE.Mesh(new THREE.CylinderGeometry(14.0*MM, 14.0*MM, 4.5*MM, 28), matHeatsink);
  frame.rotation.z = Math.PI / 2;
  g.add(frame);
  const mag = new THREE.Mesh(new THREE.CylinderGeometry(7.5*MM, 7.5*MM, 3.5*MM, 20), matAluminum);
  mag.rotation.z = Math.PI / 2;
  mag.position.x = -3.6*MM;
  g.add(mag);
  return g;
}

// --- Articulated 12-DOF Humanoid Scene Graph (Exact Match to OpenSCAD & MuJoCo!) ---
// Wrap entire OpenSCAD/MuJoCo (+Z Up) coordinate system inside Three.js (+Y Up) via -90deg X rotation!
const worldRobot = new THREE.Group();
worldRobot.rotation.x = -Math.PI / 2;
scene.add(worldRobot);

const torsoGroup = new THREE.Group();
torsoGroup.position.set(0, 0, 124.0 * MM); // Z_HIP_ROLL = 124mm
worldRobot.add(torsoGroup);

// Internal Torso Electronics (visible in X-Ray & Exploded Views!)
const batteryGroup = createBatteryPack2S();
batteryGroup.position.set(-10.0*MM, 0, 29.0*MM);
torsoGroup.add(batteryGroup);
registerExplode(batteryGroup, -36, 0, 0);

const pca9685Mesh = new THREE.Mesh(new THREE.BoxGeometry(2.2*MM, 60.0*MM, 24.0*MM), matPCBBlue);
pca9685Mesh.position.set(-2.0*MM, 0, 32.0*MM);
torsoGroup.add(pca9685Mesh);
registerExplode(pca9685Mesh, -22, 0, 8);

const imuMesh = new THREE.Mesh(new THREE.BoxGeometry(15.8*MM, 20.5*MM, 2.2*MM), matPCBBlue);
imuMesh.position.set(2.0*MM, 0, 14.0*MM); // Exact Torso CoM!
torsoGroup.add(imuMesh);
registerExplode(imuMesh, 0, 0, -14);

const buck5A = new THREE.Mesh(new THREE.BoxGeometry(18.0*MM, 28.0*MM, 6.0*MM), matPCBGreen);
buck5A.position.set(4.0*MM, -16.0*MM, 24.0*MM);
torsoGroup.add(buck5A);
registerExplode(buck5A, 16, -24, 0);

const buck2A = new THREE.Mesh(new THREE.BoxGeometry(12.0*MM, 18.0*MM, 4.5*MM), matPCBGreen);
buck2A.position.set(4.0*MM, 18.0*MM, 24.0*MM);
torsoGroup.add(buck2A);
registerExplode(buck2A, 16, 24, 0);

const cap1000uF = new THREE.Mesh(new THREE.CylinderGeometry(5.0*MM, 5.0*MM, 12.0*MM, 20), matHeatsink);
cap1000uF.position.set(11.0*MM, -18.0*MM, 19.0*MM);
torsoGroup.add(cap1000uF);
registerExplode(cap1000uF, 24, -24, -6);

const ampMax98357 = new THREE.Mesh(new THREE.BoxGeometry(2.0*MM, 18.0*MM, 19.0*MM), matPCBPurple);
ampMax98357.position.set(12.0*MM, 16.0*MM, 38.0*MM);
torsoGroup.add(ampMax98357);
registerExplode(ampMax98357, 26, 20, 8);

const speakerGroup = createSpeaker28mm();
speakerGroup.position.set(20.0*MM, 0, 34.0*MM);
torsoGroup.add(speakerGroup);
registerExplode(speakerGroup, 36, 0, 4);

// Hip Roll + Shoulder + Neck Pan Servos inside Torso
for (const sy of [-23.0, 23.0]) {
  const sv = createMG90S(false);
  sv.rotation.y = Math.PI / 2;
  sv.position.set(-2.0*MM, sy*MM, 0);
  torsoGroup.add(sv);
}
for (const side of [-1, 1]) {
  const sv = createMG90S(false);
  sv.rotation.x = side * Math.PI / 2;
  sv.position.set(0, side * 25.0*MM, 46.0*MM);
  torsoGroup.add(sv);
}
const neckPanServo = createMG90S(true);
neckPanServo.position.set(0, 0, 50.0*MM);
torsoGroup.add(neckPanServo);

// --- 2-DOF Active Spatial Head ---
const headYawJoint = new THREE.Group();
headYawJoint.position.set(0, 0, 64.0 * MM); // Global Z = 188mm
torsoGroup.add(headYawJoint);
registerExplode(headYawJoint, 0, 0, 18);

const headTiltServo = createMG90S(true);
headTiltServo.rotation.x = -Math.PI / 2;
headTiltServo.position.set(0, 0, 11.0*MM);
headYawJoint.add(headTiltServo);

const headPitchJoint = new THREE.Group();
headPitchJoint.position.set(0, 0, 16.0 * MM); // Global Z = 204mm
headYawJoint.add(headPitchJoint);
registerExplode(headPitchJoint, 0, 0, 26);

// Internal Head Electronics (XIAO ESP32S3 Sense + Coaxial VL53L0X)
const xiaoGroup = createXiaoESP32S3Sense();
xiaoGroup.position.set(11.0*MM, 0, 15.0*MM);
headPitchJoint.add(xiaoGroup);
registerExplode(xiaoGroup, 24, 0, 8);

const tofGroup = createVL53L0X();
tofGroup.position.set(19.0*MM, 0, 5.0*MM);
headPitchJoint.add(tofGroup);
registerExplode(tofGroup, 30, 0, -4);

// Active Spatial ToF Laser Beam & OV5640 Camera FOV Frustum Visualization
const sensorVizGroup = new THREE.Group();
headPitchJoint.add(sensorVizGroup);

// Camera 62-deg FOV translucent wireframe frustum expanding cleanly from lens toward +X
const camConeGeo = new THREE.ConeGeometry(0.088, 0.16, 4, 1, true);
camConeGeo.rotateZ(Math.PI / 2); // Apex at -X (lens), open square base at +X
camConeGeo.rotateX(Math.PI / 4);
const camConeWire = new THREE.LineSegments(
  new THREE.EdgesGeometry(camConeGeo),
  new THREE.LineBasicMaterial({ color: 0x0ea5e9, transparent: true, opacity: 0.26 })
);
camConeWire.position.set(0.102, 0, 15.0 * MM);
sensorVizGroup.add(camConeWire);

// Coaxial VL53L0X 940nm Laser 25-deg Scan Cone expanding cleanly outward from emitter toward +X
const tofBeamGeo = new THREE.CylinderGeometry(0.022, 0.0014, 0.17, 16, 1, true);
tofBeamGeo.rotateZ(-Math.PI / 2); // radiusBottom (1.4mm) at -X emitter, radiusTop (22mm) at +X far end!
const tofBeam = new THREE.Mesh(
  tofBeamGeo,
  new THREE.MeshBasicMaterial({ color: 0x06b6d4, transparent: true, opacity: 0.18, side: THREE.DoubleSide, depthWrite: false })
);
tofBeam.position.set(0.107, 0, 5.0 * MM);
sensorVizGroup.add(tofBeam);

const tofCenterRay = new THREE.Line(
  new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(22.0*MM, 0, 5.0*MM),
    new THREE.Vector3(192.0*MM, 0, 5.0*MM)
  ]),
  new THREE.LineBasicMaterial({ color: 0x0284c7, transparent: true, opacity: 0.55 })
);
sensorVizGroup.add(tofCenterRay);

// --- Arms ---
const lShoulderJoint = new THREE.Group();
lShoulderJoint.position.set(0, 46.0*MM, 46.0*MM);
torsoGroup.add(lShoulderJoint);
registerExplode(lShoulderJoint, 0, 32, 0);

const rShoulderJoint = new THREE.Group();
rShoulderJoint.position.set(0, -46.0*MM, 46.0*MM);
torsoGroup.add(rShoulderJoint);
registerExplode(rShoulderJoint, 0, -32, 0);

// --- Universal-Ankle Legs (L & R) ---
function buildLegChain(side) {
  const hipRoll = new THREE.Group();
  hipRoll.position.set(0, side * 23.0*MM, 0);
  torsoGroup.add(hipRoll);
  registerExplode(hipRoll, 0, side * 18, -14);

  const hipPitch = new THREE.Group();
  hipPitch.position.set(0, side * 3.0*MM, -18.0*MM);
  hipRoll.add(hipPitch);
  registerExplode(hipPitch, 0, side * 12, -16);

  // Hip Pitch MG90S inside hip_clevis
  const svHip = createMG90S(false);
  svHip.rotation.x = -Math.PI / 2;
  svHip.position.set(0, 0, -16.0*MM);
  hipRoll.add(svHip);

  const anklePitch = new THREE.Group();
  anklePitch.position.set(0, 0, -88.0*MM); // Full 88mm sculpted leg strut!
  hipPitch.add(anklePitch);
  registerExplode(anklePitch, 0, 0, -20);

  // Ankle Pitch MG90S at lower leg strut
  const svAnkP = createMG90S(false);
  svAnkP.rotation.x = -Math.PI / 2;
  svAnkP.position.set(0, 0, -82.0*MM);
  hipPitch.add(svAnkP);

  const ankleRoll = new THREE.Group();
  ankleRoll.position.set(0, 0, 0);
  anklePitch.add(ankleRoll);
  registerExplode(ankleRoll, 0, 0, -18);

  // Ankle Roll MG90S inside universal ankle_clevis
  const svAnkR = createMG90S(false);
  svAnkR.rotation.y = Math.PI / 2;
  svAnkR.position.set(0, 0, 1.5*MM);
  anklePitch.add(svAnkR);

  // 2mm Silicone Anti-Slip Sole Pad under Foot
  const pad = new THREE.Mesh(new THREE.BoxGeometry(44.0*MM, 40.0*MM, 2.0*MM), matSoleSilicone);
  pad.position.set(3.0*MM, 0, -17.0*MM);
  pad.receiveShadow = true;
  ankleRoll.add(pad);
  registerExplode(pad, 0, 0, -14);

  return { hipRoll, hipPitch, anklePitch, ankleRoll };
}

const leftLeg = buildLegChain(+1);
const rightLeg = buildLegChain(-1);

// --- Load Actual OpenSCAD-Compiled Watertight STL Parts ---
const stlLoader = new STLLoader();
function mountSTL(url, parent, mat, offsetMM = [0, 0, 0], explodeMM = null) {
  stlLoader.load(url, (geo) => {
    geo.scale(MM, MM, MM);
    geo.computeVertexNormals();
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(offsetMM[0]*MM, offsetMM[1]*MM, offsetMM[2]*MM);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    parent.add(mesh);

    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo, 26), edgeLineMat);
    edges.visible = showEdges;
    mesh.add(edges);
    edgeLines.push(edges);

    if (explodeMM) registerExplode(mesh, ...explodeMM);
  });
}

let showEdges = false;

mountSTL('cad/stl/torso_chassis.stl',    torsoGroup,        matShellPrimary, [0, 0, 0], null);
mountSTL('cad/stl/torso_back_cover.stl', torsoGroup,        matShellAccent,  [0, 0, 0], [-56, 0, 0]);
mountSTL('cad/stl/neck_gimbal.stl',      headYawJoint,      matShellAccent,  [0, 0, 0], null);
mountSTL('cad/stl/head_shell.stl',       headPitchJoint,    matShellPrimary, [0, 0, 0], null);
mountSTL('cad/stl/head_visor.stl',       headPitchJoint,    matVisor,        [0, 0, 0], [38, 0, 0]);
mountSTL('cad/stl/arm_link_left.stl',    lShoulderJoint,    matShellPrimary, [0, 0, 0], null);
mountSTL('cad/stl/arm_link_right.stl',   rShoulderJoint,    matShellPrimary, [0, 0, 0], null);

// Left Leg STLs (local origins match rnv1_params.scad to 0.01mm!)
mountSTL('cad/stl/hip_clevis_left.stl',  leftLeg.hipRoll,   matShellAccent,  [0, 3.0, -18.0], null);
mountSTL('cad/stl/leg_link.stl',         leftLeg.hipPitch,  matShellPrimary, [0, 0, -88.0],   null);
mountSTL('cad/stl/ankle_clevis.stl',     leftLeg.anklePitch,matShellAccent,  [0, 0, 0],       null);
mountSTL('cad/stl/foot_sole.stl',        leftLeg.ankleRoll, matShellPrimary, [0, 0, -18.0],   null);

// Right Leg STLs
mountSTL('cad/stl/hip_clevis_right.stl', rightLeg.hipRoll,  matShellAccent,  [0, -3.0, -18.0], null);
mountSTL('cad/stl/leg_link.stl',         rightLeg.hipPitch, matShellPrimary, [0, 0, -88.0],    null);
mountSTL('cad/stl/ankle_clevis.stl',     rightLeg.anklePitch,matShellAccent, [0, 0, 0],        null);
mountSTL('cad/stl/foot_sole.stl',        rightLeg.ankleRoll,matShellPrimary, [0, 0, -18.0],    null);

// --- Interactive UI Controls Wiring (Zero Text in Viewport) ---
let isPlaying = true;
let isXray = false;
let showSensors = true;
let explodeFactor = 0.0;
let simTime = 0.0;

const btnPlay      = document.getElementById('btn-play');
const btnXray      = document.getElementById('btn-xray');
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

function setExplode(val01) {
  explodeFactor = Math.max(0, Math.min(1, val01));
  sliderExp.value = String(Math.round(explodeFactor * 100));
  btnExplode.classList.toggle('is-active', explodeFactor > 0.02);
  if (explodeFactor > 0.02 && isPlaying) {
    isPlaying = false;
    btnPlay.classList.remove('is-active');
    resetPoseToNeutral();
  }
  for (const item of explodeEntries) {
    item.obj.position.copy(item.basePos).addScaledVector(item.explodeDir, explodeFactor);
  }
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

btnPlay.addEventListener('click', () => {
  isPlaying = !isPlaying;
  btnPlay.classList.toggle('is-active', isPlaying);
  if (isPlaying && explodeFactor > 0.0) setExplode(0.0);
});

btnXray.addEventListener('click', () => {
  isXray = !isXray;
  btnXray.classList.toggle('is-active', isXray);
  for (const m of shellMaterials) {
    m.transparent = isXray;
    m.opacity = isXray ? 0.22 : 1.0;
    m.depthWrite = !isXray;
    m.needsUpdate = true;
  }
});

btnEdges.addEventListener('click', () => {
  showEdges = !showEdges;
  btnEdges.classList.toggle('is-active', showEdges);
  for (const l of edgeLines) l.visible = showEdges;
});

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
});

btnExplode.addEventListener('click', () => {
  setExplode(explodeFactor > 0.1 ? 0.0 : 1.0);
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

// Populate Explanation Drawer Tables Dynamically from Verified Data (window.RNV1_DATA)
function populateExplanationDrawer() {
  const data = window.RNV1_DATA;
  if (!data) return;

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
      return `
        <tr>
          <td><span class="pass-tag">${s.passed ? '100% PASS' : 'FAIL'}</span></td>
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

  // Keep robot centered horizontally in studio CAD view while preserving real vertical bounce & attitude!
  torsoGroup.position.set(0, 0, lerp(a.pos[2], b.pos[2]));
  tmpQ1.set(a.quat[1], a.quat[2], a.quat[3], a.quat[0]); // MuJoCo [w,x,y,z] -> Three [x,y,z,w]
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
    camera.position.lerp(targetCamPos, 0.12);
    controls.target.lerp(targetLookAt, 0.12);
    if (camera.position.distanceTo(targetCamPos) < 0.002) camLerping = false;
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
