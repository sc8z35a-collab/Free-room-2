// RNV1 Watchmaker-Grade Ultra-Micro-Detailed 3D CAD Mechatronics Anatomy Builder
// Grounded to 0.01 mm in bom/rnv1_bom.json, bom/wiring_pinout.json, cad/rnv1_params.scad & cad/parts/*.scad.
import * as THREE from '../vendor/three/three.module.min.js';
import { createMicroComponentTextures, applySectionCapHatchShader } from './rnv1-textures.js';

export const MM = 0.001;

// --- Shared Micro-Geometries & True Extruded Involute Spur Gears ---
const GEO_PIN_GOLD     = new THREE.BoxGeometry(0.64 * MM, 0.64 * MM, 6.2 * MM);
const GEO_M2_HEAD      = new THREE.CylinderGeometry(1.9 * MM, 1.9 * MM, 1.8 * MM, 18);
const GEO_M2_HEX       = new THREE.CylinderGeometry(0.85 * MM, 0.85 * MM, 0.95 * MM, 6);
const GEO_M2_PAN       = new THREE.CylinderGeometry(1.85 * MM, 1.55 * MM, 1.2 * MM, 18);
const GEO_M2_WASHER    = new THREE.CylinderGeometry(2.35 * MM, 2.35 * MM, 0.35 * MM, 18);
const GEO_THREAD_RING  = new THREE.TorusGeometry(0.96 * MM, 0.13 * MM, 6, 14);
const GEO_SMD_0603     = new THREE.BoxGeometry(1.6 * MM, 0.8 * MM, 0.7 * MM);
const GEO_SMD_1206     = new THREE.BoxGeometry(3.2 * MM, 1.6 * MM, 1.2 * MM);
const GEO_HEATSINK_PIN = new THREE.BoxGeometry(3.5 * MM, 1.25 * MM, 1.25 * MM);

const gearGeoCache = new Map();

/**
 * Generate a true extruded involute-profile toothed spur gear BufferGeometry (axis along Z, centered).
 */
export function getSpurGearGeometry(numTeeth, pitchRadiusMM, thicknessMM, boreRadiusMM = 0.7, webHoles = false) {
  const key = `${numTeeth}_${pitchRadiusMM}_${thicknessMM}_${webHoles}`;
  if (gearGeoCache.has(key)) return gearGeoCache.get(key);

  const moduleMM = (2.0 * pitchRadiusMM) / numTeeth;
  const rAdd = (pitchRadiusMM + 0.92 * moduleMM) * MM;
  const rDed = Math.max(boreRadiusMM * 1.25, pitchRadiusMM - 1.12 * moduleMM) * MM;
  const rPitch = pitchRadiusMM * MM;

  const shape = new THREE.Shape();
  const stepAng = (Math.PI * 2.0) / numTeeth;

  for (let i = 0; i < numTeeth; ++i) {
    const a0 = i * stepAng;
    const aRoot1 = a0 + stepAng * 0.12;
    const aPitch1 = a0 + stepAng * 0.25;
    const aTip1 = a0 + stepAng * 0.34;
    const aTip2 = a0 + stepAng * 0.54;
    const aPitch2 = a0 + stepAng * 0.63;
    const aRoot2 = a0 + stepAng * 0.76;

    const pts = [
      [rDed * Math.cos(a0),      rDed * Math.sin(a0)],
      [rDed * Math.cos(aRoot1),  rDed * Math.sin(aRoot1)],
      [rPitch * Math.cos(aPitch1), rPitch * Math.sin(aPitch1)],
      [rAdd * Math.cos(aTip1),   rAdd * Math.sin(aTip1)],
      [rAdd * Math.cos(aTip2),   rAdd * Math.sin(aTip2)],
      [rPitch * Math.cos(aPitch2), rPitch * Math.sin(aPitch2)],
      [rDed * Math.cos(aRoot2),  rDed * Math.sin(aRoot2)],
    ];
    if (i === 0) shape.moveTo(pts[0][0], pts[0][1]);
    else shape.lineTo(pts[0][0], pts[0][1]);
    for (let k = 1; k < pts.length; ++k) shape.lineTo(pts[k][0], pts[k][1]);
  }
  shape.closePath();

  // Central axle bore hole
  const borePath = new THREE.Path();
  borePath.absarc(0, 0, boreRadiusMM * MM, 0, Math.PI * 2, true);
  shape.holes.push(borePath);

  // 4 circular lightening web cutouts on main output gear
  if (webHoles && pitchRadiusMM >= 4.2) {
    const rWebOrbit = pitchRadiusMM * 0.52 * MM;
    const rWebHole = pitchRadiusMM * 0.16 * MM;
    for (let w = 0; w < 4; ++w) {
      const ang = (w * Math.PI) / 2.0 + Math.PI * 0.25;
      const wp = new THREE.Path();
      wp.absarc(rWebOrbit * Math.cos(ang), rWebOrbit * Math.sin(ang), rWebHole, 0, Math.PI * 2, true);
      shape.holes.push(wp);
    }
  }

  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: thicknessMM * MM,
    bevelEnabled: true,
    bevelThickness: 0.08 * MM,
    bevelSize: 0.06 * MM,
    bevelSegments: 1,
    curveSegments: 2,
  });
  geo.translate(0, 0, -0.5 * thicknessMM * MM);
  geo.computeVertexNormals();
  gearGeoCache.set(key, geo);
  return geo;
}

/**
 * Weld duplicate STL triangle soup vertices & compute smooth creased normals
 * so rounded CAD fillets render silky-smooth while hard 30deg+ edges stay razor-sharp.
 */
export function computeCreasedNormals(stlGeo, creaseAngleDeg = 30.0) {
  const posAttr = stlGeo.attributes.position;
  if (!posAttr) return stlGeo;
  const count = posAttr.count;
  const cosThresh = Math.cos((creaseAngleDeg * Math.PI) / 180.0);

  const faceNormals = new Float32Array(count * 3);
  const vA = new THREE.Vector3(), vB = new THREE.Vector3(), vC = new THREE.Vector3();
  const cb = new THREE.Vector3(), ab = new THREE.Vector3();

  const vertMap = new Map();
  const quant = 1e5;

  for (let i = 0; i < count; i += 3) {
    vA.fromBufferAttribute(posAttr, i);
    vB.fromBufferAttribute(posAttr, i + 1);
    vC.fromBufferAttribute(posAttr, i + 2);
    cb.subVectors(vC, vB);
    ab.subVectors(vA, vB);
    cb.cross(ab).normalize();
    for (let k = 0; k < 3; ++k) {
      const idx = i + k;
      faceNormals[idx * 3]     = cb.x;
      faceNormals[idx * 3 + 1] = cb.y;
      faceNormals[idx * 3 + 2] = cb.z;
      const vx = Math.round(posAttr.getX(idx) * quant);
      const vy = Math.round(posAttr.getY(idx) * quant);
      const vz = Math.round(posAttr.getZ(idx) * quant);
      const key = `${vx},${vy},${vz}`;
      let list = vertMap.get(key);
      if (!list) {
        list = [];
        vertMap.set(key, list);
      }
      list.push(idx);
    }
  }

  const smoothNormals = new Float32Array(count * 3);
  for (const indices of vertMap.values()) {
    const len = indices.length;
    for (let a = 0; a < len; ++a) {
      const ia = indices[a] * 3;
      const nx0 = faceNormals[ia], ny0 = faceNormals[ia + 1], nz0 = faceNormals[ia + 2];
      let sx = nx0, sy = ny0, sz = nz0;
      for (let b = 0; b < len; ++b) {
        if (a === b) continue;
        const ib = indices[b] * 3;
        const dot = nx0 * faceNormals[ib] + ny0 * faceNormals[ib + 1] + nz0 * faceNormals[ib + 2];
        if (dot >= cosThresh) {
          sx += faceNormals[ib];
          sy += faceNormals[ib + 1];
          sz += faceNormals[ib + 2];
        }
      }
      const inv = 1.0 / Math.hypot(sx, sy, sz || 1e-9);
      smoothNormals[ia]     = sx * inv;
      smoothNormals[ia + 1] = sy * inv;
      smoothNormals[ia + 2] = sz * inv;
    }
  }

  stlGeo.setAttribute('normal', new THREE.BufferAttribute(smoothNormals, 3));
  return stlGeo;
}

/**
 * Create curated PBR + Silkscreened PCB + Cross-Section Hatched CAD Materials.
 */
export function createAnatomyMaterials(sectionPlanes, maxAnisotropy = 8) {
  const tex = createMicroComponentTextures(maxAnisotropy);
  const hatchUniform = { value: 0.0 };

  const shellPrimary = new THREE.MeshStandardMaterial({
    color: 0xf4f6f9, roughness: 0.38, metalness: 0.05,
    side: THREE.DoubleSide, clippingPlanes: sectionPlanes, clipShadows: true,
  });
  const shellAccent = new THREE.MeshStandardMaterial({
    color: 0xe2e7f0, roughness: 0.44, metalness: 0.08,
    side: THREE.DoubleSide, clippingPlanes: sectionPlanes, clipShadows: true,
  });
  const servoMG90S = new THREE.MeshStandardMaterial({
    color: 0x1b4785, roughness: 0.34, metalness: 0.12,
    side: THREE.DoubleSide, clippingPlanes: sectionPlanes, clipShadows: true,
  });

  // Apply 45-degree engineering CAD cross-hatch shader on sliced watertight solid walls!
  applySectionCapHatchShader(shellPrimary, hatchUniform, [0.83, 0.88, 0.94], [0.05, 0.44, 0.72]);
  applySectionCapHatchShader(shellAccent,  hatchUniform, [0.78, 0.84, 0.92], [0.05, 0.44, 0.72]);
  applySectionCapHatchShader(servoMG90S,   hatchUniform, [0.22, 0.42, 0.72], [0.85, 0.92, 0.98]);

  return {
    hatchUniform,
    shellPrimary,
    shellAccent,
    visor: new THREE.MeshPhysicalMaterial({
      color: 0x121722, roughness: 0.10, metalness: 0.25, clearcoat: 1.0, clearcoatRoughness: 0.08,
      side: THREE.DoubleSide, clippingPlanes: sectionPlanes, clipShadows: true,
    }),
    servoMG90S,
    servoSG90: new THREE.MeshPhysicalMaterial({
      color: 0x2979cf, roughness: 0.22, metalness: 0.05,
      transparent: true, opacity: 0.70, clearcoat: 0.6,
      side: THREE.DoubleSide, clippingPlanes: sectionPlanes, clipShadows: true,
    }),
    rfShieldGhostable: new THREE.MeshStandardMaterial({
      color: 0xd8dfe8, roughness: 0.22, metalness: 0.88,
      side: THREE.DoubleSide, clippingPlanes: sectionPlanes,
    }),
    camBarrelClip: new THREE.MeshStandardMaterial({
      color: 0x1b1f26, roughness: 0.30, metalness: 0.78,
      side: THREE.DoubleSide, clippingPlanes: sectionPlanes,
    }),

    // Silkscreened & Laser-Marked Component Materials
    pcbPCA9685:   new THREE.MeshStandardMaterial({ map: tex.pca9685Tex,   roughness: 0.38, metalness: 0.16 }),
    pcbMPU6050:   new THREE.MeshStandardMaterial({ map: tex.mpu6050Tex,   roughness: 0.38, metalness: 0.16 }),
    pcbXIAO:      new THREE.MeshStandardMaterial({ map: tex.xiaoPcbTex,   roughness: 0.40, metalness: 0.16 }),
    pcbBuck:      new THREE.MeshStandardMaterial({ map: tex.buckPcbTex,   roughness: 0.40, metalness: 0.18 }),
    pcbBMS:       new THREE.MeshStandardMaterial({ map: tex.bmsPcbTex,    roughness: 0.40, metalness: 0.18 }),
    pcbPurple:    new THREE.MeshStandardMaterial({ map: tex.purplePcbTex, roughness: 0.40, metalness: 0.15 }),
    liIonBlue:    new THREE.MeshStandardMaterial({ map: tex.liIonCellTex, roughness: 0.26, metalness: 0.34 }),
    capSleeve:    new THREE.MeshStandardMaterial({ map: tex.capSleeveTex, roughness: 0.30, metalness: 0.24 }),
    ferrite4R7:   new THREE.MeshStandardMaterial({ map: tex.inductor4R7Tex, roughness: 0.52, metalness: 0.24 }),
    icLaser:      new THREE.MeshStandardMaterial({ map: tex.icLaserTex,   roughness: 0.44, metalness: 0.18 }),

    // Unclipped Internal Precision Mechatronics Materials
    soleSilicone:  new THREE.MeshStandardMaterial({ color: 0x21252c, roughness: 0.88, metalness: 0.02 }),
    stainless:     new THREE.MeshStandardMaterial({ color: 0xd8dfe8, roughness: 0.20, metalness: 0.90 }),
    darkSteel:     new THREE.MeshStandardMaterial({ color: 0x38404c, roughness: 0.30, metalness: 0.82 }),
    brass:         new THREE.MeshStandardMaterial({ color: 0xd6a848, roughness: 0.22, metalness: 0.88 }),
    gold:          new THREE.MeshStandardMaterial({ color: 0xefb93c, roughness: 0.16, metalness: 0.94 }),
    copper:        new THREE.MeshStandardMaterial({ color: 0xc96b38, roughness: 0.24, metalness: 0.88 }),
    nomexSpider:   new THREE.MeshStandardMaterial({ color: 0xb47b2a, roughness: 0.62, metalness: 0.08, side: THREE.DoubleSide }),
    timPadBlue:    new THREE.MeshStandardMaterial({ color: 0x38bdf8, roughness: 0.65, metalness: 0.02 }),
    cmosWafer:     new THREE.MeshPhysicalMaterial({ color: 0x059669, roughness: 0.05, metalness: 0.65, clearcoat: 1.0, iridescence: 1.0 }),
    irCutFilter:   new THREE.MeshPhysicalMaterial({ color: 0xbe185d, roughness: 0.06, metalness: 0.10, transparent: true, opacity: 0.52 }),
    anodizedBlack: new THREE.MeshStandardMaterial({ color: 0x1b1f26, roughness: 0.32, metalness: 0.76 }),
    aluminum:      new THREE.MeshStandardMaterial({ color: 0xcfd6e0, roughness: 0.24, metalness: 0.86 }),
    nylonWhite:    new THREE.MeshStandardMaterial({ color: 0xf2f5f9, roughness: 0.36, metalness: 0.03 }),
    pcbGreenPlain: new THREE.MeshStandardMaterial({ color: 0x15653c, roughness: 0.44, metalness: 0.12 }),
    pcbBlackPlain: new THREE.MeshStandardMaterial({ color: 0x171b22, roughness: 0.44, metalness: 0.12 }),
    icEpoxy:       new THREE.MeshStandardMaterial({ color: 0x14171c, roughness: 0.48, metalness: 0.15 }),
    ceramicCap:    new THREE.MeshStandardMaterial({ color: 0xb58852, roughness: 0.38, metalness: 0.10 }),
    termGreen:     new THREE.MeshStandardMaterial({ color: 0x1f8b4c, roughness: 0.40, metalness: 0.05 }),
    hdrYellow:     new THREE.MeshStandardMaterial({ color: 0xeab308, roughness: 0.42, metalness: 0.04 }),
    hdrRed:        new THREE.MeshStandardMaterial({ color: 0xdc2626, roughness: 0.42, metalness: 0.04 }),
    hdrBlack:      new THREE.MeshStandardMaterial({ color: 0x181b20, roughness: 0.48, metalness: 0.04 }),
    fpcGold:       new THREE.MeshStandardMaterial({ color: 0xd98a2b, roughness: 0.34, metalness: 0.45, side: THREE.DoubleSide }),
    speakerCone:   new THREE.MeshStandardMaterial({ color: 0x1e2229, roughness: 0.72, metalness: 0.06, side: THREE.DoubleSide }),
    lensGlass:     new THREE.MeshPhysicalMaterial({ color: 0x0c345a, roughness: 0.04, metalness: 0.08, transparent: true, opacity: 0.76, clearcoat: 1.0 }),
    tofWindow:     new THREE.MeshPhysicalMaterial({ color: 0x2b1222, roughness: 0.08, metalness: 0.20, clearcoat: 1.0 }),

    // Color-Coded Silicone Wire Insulation Palette
    wireVbatRed:   new THREE.MeshStandardMaterial({ color: 0xdc2626, roughness: 0.50, metalness: 0.04 }),
    wireGndBlack:  new THREE.MeshStandardMaterial({ color: 0x1f242d, roughness: 0.54, metalness: 0.04 }),
    wireServo5v6:  new THREE.MeshStandardMaterial({ color: 0xea580c, roughness: 0.48, metalness: 0.04 }),
    wireLogic5v0:  new THREE.MeshStandardMaterial({ color: 0x10b981, roughness: 0.48, metalness: 0.04 }),
    wireI2cSda:    new THREE.MeshStandardMaterial({ color: 0x0284c7, roughness: 0.46, metalness: 0.04 }),
    wireI2cScl:    new THREE.MeshStandardMaterial({ color: 0xd97706, roughness: 0.46, metalness: 0.04 }),
    wireI2sAudio:  new THREE.MeshStandardMaterial({ color: 0x7c3aed, roughness: 0.46, metalness: 0.04 }),
    wirePwmSig:    new THREE.MeshStandardMaterial({ color: 0xf59e0b, roughness: 0.48, metalness: 0.04 }),
    wireSenseTeal: new THREE.MeshStandardMaterial({ color: 0x06b6d4, roughness: 0.48, metalness: 0.04 }),
  };
}

function addThreadRidges(group, mats, shaftLenMM) {
  const nRings = Math.max(3, Math.floor(shaftLenMM / 0.85));
  for (let i = 0; i < nRings; ++i) {
    const ring = new THREE.Mesh(GEO_THREAD_RING, mats.darkSteel);
    ring.position.z = -(0.8 + i * 0.8) * MM;
    group.add(ring);
  }
}

/**
 * DIN 912 M2 Stainless Socket-Head Cap Screw + Flat DIN 125 Washer + Machined Thread Ridges.
 */
export function createM2SocketBolt(mats, shaftLenMM = 6.0) {
  const g = new THREE.Group();
  const washer = new THREE.Mesh(GEO_M2_WASHER, mats.stainless);
  washer.rotation.x = Math.PI / 2;
  washer.position.z = 0.18 * MM;
  g.add(washer);

  const head = new THREE.Mesh(GEO_M2_HEAD, mats.stainless);
  head.rotation.x = Math.PI / 2;
  head.position.z = 1.25 * MM;
  g.add(head);

  const hex = new THREE.Mesh(GEO_M2_HEX, mats.darkSteel);
  hex.rotation.x = Math.PI / 2;
  hex.position.z = 1.75 * MM;
  g.add(hex);

  const shaft = new THREE.Mesh(
    new THREE.CylinderGeometry(0.95 * MM, 0.92 * MM, shaftLenMM * MM, 12),
    mats.stainless
  );
  shaft.rotation.x = Math.PI / 2;
  shaft.position.z = -(shaftLenMM * 0.5) * MM;
  g.add(shaft);
  addThreadRidges(g, mats, shaftLenMM);
  return g;
}

/**
 * M2 Pan-Head Cross-Recessed Screw + Flat Washer + Helical Thread Ridges.
 */
export function createM2PanScrew(mats, shaftLenMM = 5.0) {
  const g = new THREE.Group();
  const washer = new THREE.Mesh(GEO_M2_WASHER, mats.stainless);
  washer.rotation.x = Math.PI / 2;
  washer.position.z = 0.16 * MM;
  g.add(washer);

  const head = new THREE.Mesh(GEO_M2_PAN, mats.stainless);
  head.rotation.x = Math.PI / 2;
  head.position.z = 0.9 * MM;
  g.add(head);

  const slot1 = new THREE.Mesh(new THREE.BoxGeometry(2.1 * MM, 0.42 * MM, 0.35 * MM), mats.darkSteel);
  slot1.position.z = 1.45 * MM;
  g.add(slot1);
  const slot2 = new THREE.Mesh(new THREE.BoxGeometry(0.42 * MM, 2.1 * MM, 0.35 * MM), mats.darkSteel);
  slot2.position.z = 1.45 * MM;
  g.add(slot2);

  const shaft = new THREE.Mesh(
    new THREE.CylinderGeometry(0.92 * MM, 0.82 * MM, shaftLenMM * MM, 10),
    mats.stainless
  );
  shaft.rotation.x = Math.PI / 2;
  shaft.position.z = -(shaftLenMM * 0.5) * MM;
  g.add(shaft);
  addThreadRidges(g, mats, Math.min(shaftLenMM, 4.5));
  return g;
}

/**
 * Opposing Coaxial Ø4.0 mm Dual-Shear Flanged Brass Sleeve Bushing + Retaining Circlip & Pin.
 */
export function createDualShearBushing(mats) {
  const g = new THREE.Group();
  const sleeve = new THREE.Mesh(new THREE.CylinderGeometry(2.0 * MM, 2.0 * MM, 3.8 * MM, 20), mats.brass);
  sleeve.rotation.x = Math.PI / 2;
  sleeve.position.z = -1.9 * MM;
  g.add(sleeve);

  const flange = new THREE.Mesh(new THREE.CylinderGeometry(3.4 * MM, 3.4 * MM, 0.75 * MM, 20), mats.brass);
  flange.rotation.x = Math.PI / 2;
  flange.position.z = 0.38 * MM;
  g.add(flange);

  const circlip = new THREE.Mesh(new THREE.TorusGeometry(2.1 * MM, 0.28 * MM, 8, 18, Math.PI * 1.65), mats.darkSteel);
  circlip.position.z = 0.85 * MM;
  g.add(circlip);

  const pin = new THREE.Mesh(new THREE.CylinderGeometry(1.35 * MM, 1.35 * MM, 4.8 * MM, 16), mats.stainless);
  pin.rotation.x = Math.PI / 2;
  pin.position.z = -1.5 * MM;
  g.add(pin);
  return g;
}

/**
 * Molded Servo Horn + Central M2x5 Retaining Screw + Radial Locking Screws.
 */
export function createServoHornAssembly(mats, { isSG90 = false, armLenMM = 12.0, doubleArm = false } = {}) {
  const g = new THREE.Group();
  const hornMat = isSG90 ? mats.nylonWhite : mats.anodizedBlack;

  const hub = new THREE.Mesh(new THREE.CylinderGeometry(3.55 * MM, 3.55 * MM, 3.2 * MM, 20), hornMat);
  hub.rotation.x = Math.PI / 2;
  hub.position.z = 0.4 * MM;
  g.add(hub);

  const arms = doubleArm ? [-1, 1] : [1];
  for (const dir of arms) {
    const arm = new THREE.Mesh(new THREE.BoxGeometry(armLenMM * MM, 4.6 * MM, 1.85 * MM), hornMat);
    arm.position.set(dir * (armLenMM * 0.5) * MM, 0, 1.1 * MM);
    g.add(arm);

    const tip = new THREE.Mesh(new THREE.CylinderGeometry(2.15 * MM, 2.15 * MM, 1.85 * MM, 16), hornMat);
    tip.rotation.x = Math.PI / 2;
    tip.position.set(dir * armLenMM * MM, 0, 1.1 * MM);
    g.add(tip);

    const rScrew = createM2PanScrew(mats, 3.6);
    rScrew.position.set(dir * (armLenMM - 2.2) * MM, 0, 2.0 * MM);
    g.add(rScrew);
  }

  const centerScrew = createM2PanScrew(mats, 4.8);
  centerScrew.position.z = 2.1 * MM;
  g.add(centerScrew);

  return { group: g, centerScrew };
}

/**
 * Watchmaker-Grade MG90S / SG90 Micro Servo with TRUE INVOLUTE SPUR GEAR TRAIN & LIVE KINEMATICS!
 *   - Output 21T serrated spline at (0, 0, 0) along +Z
 *   - 4 meshing toothed gear stages (9T Pinion -> 18T Stage1 -> 22T Stage2 -> 28T Webbed Output Gear)
 *   - Exposes `root.userData.updateGearTrain(jointAngleRad)` so gears spin live during walking!
 */
export function createPrecisionServo(mats, isSG90 = false) {
  const root = new THREE.Group();
  const caseMat = isSG90 ? mats.servoSG90 : mats.servoMG90S;
  const gearMat = isSG90 ? mats.nylonWhite : mats.brass;
  const bodyL = (isSG90 ? 22.5 : 22.8) * MM;
  const bodyW = (isSG90 ? 12.0 : 12.2) * MM;
  const bodyH = (isSG90 ? 19.5 : 21.5) * MM;
  const xOff = -5.5 * MM;

  // 1. Outer Housing (with case split seam & 4 bottom M1.7 tie bolts)
  const housingGroup = new THREE.Group();
  root.add(housingGroup);

  const mainCase = new THREE.Mesh(new THREE.BoxGeometry(bodyL, bodyW, bodyH), caseMat);
  mainCase.position.set(xOff, 0, -3.5 * MM - bodyH * 0.5);
  mainCase.castShadow = true;
  housingGroup.add(mainCase);

  const topCap = new THREE.Mesh(new THREE.BoxGeometry(bodyL, bodyW, 3.4 * MM), caseMat);
  topCap.position.set(xOff, 0, -1.8 * MM);
  housingGroup.add(topCap);

  const bearingDome = new THREE.Mesh(
    new THREE.CylinderGeometry(5.8 * MM, 6.0 * MM, 3.2 * MM, 24),
    isSG90 ? caseMat : mats.aluminum
  );
  bearingDome.rotation.x = Math.PI / 2;
  bearingDome.position.set(0, 0, -1.2 * MM);
  housingGroup.add(bearingDome);

  // Exposed Sealed Ball Bearing Race at Output Shaft!
  const ballBearing = new THREE.Mesh(new THREE.TorusGeometry(3.2 * MM, 0.65 * MM, 10, 22), mats.stainless);
  ballBearing.position.set(0, 0, -0.1 * MM);
  root.add(ballBearing);

  const flange = new THREE.Mesh(new THREE.BoxGeometry(32.2 * MM, bodyW, 2.4 * MM), caseMat);
  flange.position.set(xOff, 0, -8.2 * MM);
  housingGroup.add(flange);

  for (const dx of [-13.8, 13.8]) {
    const eyelet = new THREE.Mesh(new THREE.CylinderGeometry(1.8 * MM, 1.8 * MM, 2.6 * MM, 14), mats.brass);
    eyelet.rotation.x = Math.PI / 2;
    eyelet.position.set(xOff + dx * MM, 0, -8.2 * MM);
    root.add(eyelet);

    const mScrew = createM2PanScrew(mats, 5.5);
    mScrew.position.set(xOff + dx * MM, 0, -6.9 * MM);
    root.add(mScrew);
  }

  // 4x Bottom Case Assembly Tie-Bolts
  for (const cx of [-10.0, 10.0]) {
    for (const cy of [-4.8, 4.8]) {
      const cBolt = new THREE.Mesh(new THREE.CylinderGeometry(0.8 * MM, 0.8 * MM, 0.8 * MM, 10), mats.stainless);
      cBolt.rotation.x = Math.PI / 2;
      cBolt.position.set(xOff + cx * MM, cy * MM, -3.5 * MM - bodyH - 0.3 * MM);
      root.add(cBolt);
    }
  }

  // 2. True 21-Tooth Serrated Output Spline (rotates with output gear!)
  const splineGeo = getSpurGearGeometry(21, 2.35, 3.4, 0.85, false);
  const splineMesh = new THREE.Mesh(splineGeo, isSG90 ? mats.nylonWhite : mats.brass);
  splineMesh.position.set(0, 0, 1.65 * MM);
  root.add(splineMesh);

  // 3. Internal Anatomy: Coreless DC Motor + 4-Stage Toothed Involute Spur Gear Train!
  const internals = new THREE.Group();
  root.add(internals);

  const motorCan = new THREE.Mesh(new THREE.CylinderGeometry(4.8 * MM, 4.8 * MM, 13.2 * MM, 22), mats.stainless);
  motorCan.rotation.x = Math.PI / 2;
  motorCan.position.set(-10.6 * MM, 0, -15.5 * MM);
  internals.add(motorCan);

  // Internal copper rotor armature windings visible at motor top vent
  const copperRotor = new THREE.Mesh(new THREE.CylinderGeometry(3.8 * MM, 3.8 * MM, 2.2 * MM, 18), mats.copper);
  copperRotor.rotation.x = Math.PI / 2;
  copperRotor.position.set(-10.6 * MM, 0, -8.9 * MM);
  internals.add(copperRotor);

  const motorEndbell = new THREE.Mesh(new THREE.CylinderGeometry(4.6 * MM, 4.6 * MM, 1.6 * MM, 18), mats.copper);
  motorEndbell.rotation.x = Math.PI / 2;
  motorEndbell.position.set(-10.6 * MM, 0, -22.2 * MM);
  internals.add(motorEndbell);

  // Stage 0: 9T High-Speed Motor Brass Pinion
  const gPinion = new THREE.Mesh(getSpurGearGeometry(9, 1.35, 3.0, 0.45, false), mats.brass);
  gPinion.position.set(-10.6 * MM, 0, -7.0 * MM);
  internals.add(gPinion);

  // Stage 1: 18T Intermediate Spur + 9T Compound Pinion
  const gStage1 = new THREE.Group();
  gStage1.position.set(-7.9 * MM, 1.6 * MM, -6.5 * MM);
  gStage1.add(new THREE.Mesh(getSpurGearGeometry(18, 3.2, 1.4, 0.55, false), gearMat));
  const s1Pin = new THREE.Mesh(getSpurGearGeometry(9, 1.3, 2.8, 0.50, false), mats.darkSteel);
  s1Pin.position.z = 1.2 * MM;
  gStage1.add(s1Pin);
  internals.add(gStage1);

  // Stage 2: 22T Pre-Output Spur + 10T Compound Pinion
  const gStage2 = new THREE.Group();
  gStage2.position.set(-4.9 * MM, -1.5 * MM, -5.0 * MM);
  gStage2.add(new THREE.Mesh(getSpurGearGeometry(22, 3.8, 1.5, 0.60, false), gearMat));
  const s2Pin = new THREE.Mesh(getSpurGearGeometry(10, 1.5, 2.6, 0.55, false), mats.darkSteel);
  s2Pin.position.z = 1.3 * MM;
  gStage2.add(s2Pin);
  internals.add(gStage2);

  // Stage 3: 28T Webbed Main Output Spur Gear concentric with shaft (0, 0, -3.3 mm)!
  const gOutput = new THREE.Mesh(getSpurGearGeometry(28, 4.9, 1.7, 1.1, true), gearMat);
  gOutput.position.set(0, 0, -3.3 * MM);
  internals.add(gOutput);

  // Stainless Ground Gear Axles
  for (const ax of [[-7.9, 1.6], [-4.9, -1.5]]) {
    const pinAxle = new THREE.Mesh(new THREE.CylinderGeometry(0.55 * MM, 0.55 * MM, 6.2 * MM, 10), mats.stainless);
    pinAxle.rotation.x = Math.PI / 2;
    pinAxle.position.set(ax[0] * MM, ax[1] * MM, -5.2 * MM);
    internals.add(pinAxle);
  }

  // Rotary Feedback Potentiometer + Wiper Shaft + H-Bridge Servo Control PCB
  const potBody = new THREE.Mesh(new THREE.CylinderGeometry(3.8 * MM, 3.8 * MM, 4.6 * MM, 18), mats.anodizedBlack);
  potBody.rotation.x = Math.PI / 2;
  potBody.position.set(0, 0, -10.2 * MM);
  internals.add(potBody);

  const ctrlPcb = new THREE.Mesh(new THREE.BoxGeometry(9.6 * MM, 9.8 * MM, 1.2 * MM), mats.pcbBuck);
  ctrlPcb.position.set(-1.2 * MM, 0, -19.5 * MM);
  internals.add(ctrlPcb);
  const driverIc = new THREE.Mesh(new THREE.BoxGeometry(3.8 * MM, 3.2 * MM, 1.0 * MM), mats.icLaser);
  driverIc.position.set(-1.2 * MM, 0, -18.5 * MM);
  internals.add(driverIc);

  const boot = new THREE.Mesh(new THREE.BoxGeometry(3.2 * MM, 4.4 * MM, 2.8 * MM), mats.soleSilicone);
  boot.position.set(xOff - bodyL * 0.5 - 1.2 * MM, 0, -21.0 * MM);
  internals.add(boot);

  // Live Meshing Gear Ratio Hook!
  root.userData.updateGearTrain = (thetaRad) => {
    splineMesh.rotation.z = thetaRad;
    gOutput.rotation.z    = thetaRad;
    gStage2.rotation.z    = -thetaRad * 3.8 + 0.15;
    gStage1.rotation.z    =  thetaRad * 15.2;
    gPinion.rotation.z    = -thetaRad * 62.0;
    copperRotor.rotation.y = -thetaRad * 62.0;
  };

  return root;
}

/**
 * Seeed Studio XIAO ESP32S3 Sense with Under-Shield Silicon SoC, Crystal, TIM Pad,
 * 25-Pin CNC Heatsink, Silkscreened PCB, EPDM Mic Gasket, and 3-Element OV5640 Optical Stack!
 */
export function createXiaoESP32S3SenseAssembly(mats) {
  const g = new THREE.Group();

  // 1. Main Silkscreened XIAO ESP32S3 4-Layer PCB
  const mainPcb = new THREE.Mesh(new THREE.BoxGeometry(1.6 * MM, 17.8 * MM, 21.0 * MM), mats.pcbXIAO);
  g.add(mainPcb);

  // 14x Gold Castellated Half-Holes
  for (const side of [-1, 1]) {
    for (let i = -3; i <= 3; ++i) {
      const pad = new THREE.Mesh(new THREE.CylinderGeometry(0.85 * MM, 0.85 * MM, 1.75 * MM, 12), mats.gold);
      pad.rotation.z = Math.PI / 2;
      pad.position.set(0, side * 8.6 * MM, i * 2.54 * MM);
      g.add(pad);
    }
  }

  // Top Rolled Stainless USB Type-C Receptacle + 2x Micro Tactile Reset/Boot Buttons
  const usbShell = new THREE.Mesh(new THREE.BoxGeometry(3.2 * MM, 8.94 * MM, 7.3 * MM), mats.stainless);
  usbShell.position.set(2.2 * MM, 0, 8.2 * MM);
  g.add(usbShell);
  const usbTongue = new THREE.Mesh(new THREE.BoxGeometry(0.7 * MM, 6.6 * MM, 4.0 * MM), mats.icEpoxy);
  usbTongue.position.set(2.2 * MM, 0, 9.8 * MM);
  g.add(usbTongue);

  for (const sy of [-6.2, 6.2]) {
    const btn = new THREE.Mesh(new THREE.BoxGeometry(1.1 * MM, 2.6 * MM, 1.8 * MM), mats.gold);
    btn.position.set(1.2 * MM, sy * MM, 8.2 * MM);
    g.add(btn);
  }

  // UNDER-SHIELD SILICON ANATOMY (revealed when RF shield turns translucent in X-Ray / Section mode!):
  // Espressif ESP32-S3R8 Dual-Core QFN-56 SoC + 8MB SPI NOR Flash + 40MHz Crystal Oscillator!
  const espSoc = new THREE.Mesh(new THREE.BoxGeometry(0.95 * MM, 7.0 * MM, 7.0 * MM), mats.icLaser);
  espSoc.position.set(-1.3 * MM, 0, -1.2 * MM);
  g.add(espSoc);
  const flashIc = new THREE.Mesh(new THREE.BoxGeometry(0.85 * MM, 4.0 * MM, 3.0 * MM), mats.icEpoxy);
  flashIc.position.set(-1.25 * MM, -2.5 * MM, -5.8 * MM);
  g.add(flashIc);
  const xtal40M = new THREE.Mesh(new THREE.BoxGeometry(0.75 * MM, 2.5 * MM, 3.2 * MM), mats.gold);
  xtal40M.position.set(-1.2 * MM, 3.2 * MM, -5.8 * MM);
  g.add(xtal40M);

  // Nickel RF Shield Can + Sky-Blue 0.5mm Silicone Thermal Interface Pad (TIM) + 25-Pin CNC Heatsink!
  const rfShield = new THREE.Mesh(new THREE.BoxGeometry(2.2 * MM, 12.2 * MM, 12.2 * MM), mats.rfShieldGhostable);
  rfShield.position.set(-1.9 * MM, 0, -1.5 * MM);
  g.add(rfShield);

  const timPad = new THREE.Mesh(new THREE.BoxGeometry(0.55 * MM, 12.0 * MM, 12.0 * MM), mats.timPadBlue);
  timPad.position.set(-3.25 * MM, 0, -1.5 * MM);
  g.add(timPad);

  const hsBase = new THREE.Mesh(new THREE.BoxGeometry(1.5 * MM, 13.6 * MM, 13.6 * MM), mats.anodizedBlack);
  hsBase.position.set(-4.2 * MM, 0, -1.5 * MM);
  g.add(hsBase);

  for (let iy = -2; iy <= 2; ++iy) {
    for (let iz = -2; iz <= 2; ++iz) {
      const fin = new THREE.Mesh(GEO_HEATSINK_PIN, mats.anodizedBlack);
      fin.position.set(-6.6 * MM, iy * 2.6 * MM, (-1.5 + iz * 2.6) * MM);
      g.add(fin);
    }
  }

  // 2. B2B Mezzanine Connector + Sense Expansion Daughterboard
  const b2b = new THREE.Mesh(new THREE.BoxGeometry(2.8 * MM, 10.0 * MM, 4.2 * MM), mats.icEpoxy);
  b2b.position.set(2.2 * MM, 0, -4.5 * MM);
  g.add(b2b);

  const sensePcb = new THREE.Mesh(new THREE.BoxGeometry(1.6 * MM, 17.8 * MM, 21.0 * MM), mats.pcbXIAO);
  sensePcb.position.set(4.4 * MM, 0, 0);
  g.add(sensePcb);

  // MSM261 PDM Digital Mic + Acoustic EPDM Isolation Foam Seal Ring
  const pdmMic = new THREE.Mesh(new THREE.BoxGeometry(1.1 * MM, 3.5 * MM, 2.65 * MM), mats.stainless);
  pdmMic.position.set(5.6 * MM, 0, 7.6 * MM);
  g.add(pdmMic);
  const micGasket = new THREE.Mesh(new THREE.TorusGeometry(2.2 * MM, 0.65 * MM, 10, 20), mats.soleSilicone);
  micGasket.rotation.y = Math.PI / 2;
  micGasket.position.set(6.3 * MM, 0, 7.6 * MM);
  g.add(micGasket);

  const sdSlot = new THREE.Mesh(new THREE.BoxGeometry(1.8 * MM, 11.5 * MM, 11.0 * MM), mats.stainless);
  sdSlot.position.set(2.8 * MM, 0, 2.0 * MM);
  g.add(sdSlot);

  // 3. OV5640 / OV3660 Internal Optical Cross-Section:
  // Iridescent CMOS Wafer Die + 650nm IR-Cut Filter + 3 Stacked Optical Glass Elements inside M7 Barrel!
  const fpcRibbon = new THREE.Mesh(new THREE.BoxGeometry(0.35 * MM, 8.0 * MM, 11.5 * MM), mats.fpcGold);
  fpcRibbon.position.set(5.5 * MM, 0, -2.5 * MM);
  g.add(fpcRibbon);

  const camBase = new THREE.Mesh(new THREE.BoxGeometry(3.2 * MM, 8.5 * MM, 8.5 * MM), mats.camBarrelClip);
  camBase.position.set(6.8 * MM, 0, 0);
  g.add(camBase);

  const cmosDie = new THREE.Mesh(new THREE.BoxGeometry(0.4 * MM, 5.2 * MM, 4.0 * MM), mats.cmosWafer);
  cmosDie.position.set(6.0 * MM, 0, 0);
  g.add(cmosDie);

  const irFilter = new THREE.Mesh(new THREE.BoxGeometry(0.35 * MM, 5.8 * MM, 5.8 * MM), mats.irCutFilter);
  irFilter.position.set(7.4 * MM, 0, 0);
  g.add(irFilter);

  const lensBarrel = new THREE.Mesh(new THREE.CylinderGeometry(4.0 * MM, 4.2 * MM, 4.6 * MM, 28), mats.camBarrelClip);
  lensBarrel.rotation.z = Math.PI / 2;
  lensBarrel.position.set(9.4 * MM, 0, 0);
  g.add(lensBarrel);

  // 3 internal multi-coated optical glass elements along optical axis X = 8.4, 9.6, 10.9 mm!
  for (const [lx, lr] of [[8.4, 2.3], [9.6, 2.55], [10.9, 2.85]]) {
    const el = new THREE.Mesh(
      new THREE.SphereGeometry(lr * MM, 22, 14, 0, Math.PI * 2, 0, Math.PI * 0.45),
      mats.lensGlass
    );
    el.rotation.z = -Math.PI / 2;
    el.position.set(lx * MM, 0, 0);
    g.add(el);
  }

  const lensBezel = new THREE.Mesh(new THREE.TorusGeometry(3.3 * MM, 0.45 * MM, 12, 24), mats.aluminum);
  lensBezel.rotation.y = Math.PI / 2;
  lensBezel.position.set(11.6 * MM, 0, 0);
  g.add(lensBezel);

  // 4. U.FL Coax + 2.4 GHz Wi-Fi/BLE FPC Dipole Antenna
  const ufl = new THREE.Mesh(new THREE.CylinderGeometry(1.1 * MM, 1.1 * MM, 1.0 * MM, 12), mats.gold);
  ufl.rotation.z = Math.PI / 2;
  ufl.position.set(-1.1 * MM, 6.2 * MM, -7.8 * MM);
  g.add(ufl);

  const fpcAnt = new THREE.Mesh(new THREE.BoxGeometry(8.5 * MM, 0.4 * MM, 20.0 * MM), mats.fpcGold);
  fpcAnt.position.set(-2.0 * MM, 14.2 * MM, -1.0 * MM);
  g.add(fpcAnt);

  return g;
}

/**
 * VL53L0X (GY-530) 940 nm Laser Time-of-Flight Sensor with Silkscreened Purple PCB & Dual Cavities.
 */
export function createVL53L0XAssembly(mats) {
  const g = new THREE.Group();
  const pcb = new THREE.Mesh(new THREE.BoxGeometry(1.6 * MM, 13.3 * MM, 10.5 * MM), mats.pcbPurple);
  g.add(pcb);

  for (const sy of [-4.8, 4.8]) {
    const pad = new THREE.Mesh(new THREE.CylinderGeometry(1.6 * MM, 1.6 * MM, 1.7 * MM, 14), mats.gold);
    pad.rotation.z = Math.PI / 2;
    pad.position.set(0, sy * MM, -3.2 * MM);
    g.add(pad);
  }

  const lgaCap = new THREE.Mesh(new THREE.BoxGeometry(1.2 * MM, 4.4 * MM, 2.4 * MM), mats.icEpoxy);
  lgaCap.position.set(1.3 * MM, 0, 0);
  g.add(lgaCap);

  // Gold optical barrier septum between 940nm VCSEL emitter & SPAD receiver
  const septum = new THREE.Mesh(new THREE.BoxGeometry(0.35 * MM, 0.45 * MM, 2.2 * MM), mats.gold);
  septum.position.set(1.85 * MM, 0, 0);
  g.add(septum);

  for (const dy of [-1.1, 1.1]) {
    const optic = new THREE.Mesh(new THREE.CylinderGeometry(0.68 * MM, 0.68 * MM, 0.4 * MM, 16), mats.tofWindow);
    optic.rotation.z = Math.PI / 2;
    optic.position.set(1.85 * MM, dy * MM, 0);
    g.add(optic);
  }

  const ldo = new THREE.Mesh(new THREE.BoxGeometry(1.0 * MM, 2.9 * MM, 1.6 * MM), mats.icLaser);
  ldo.position.set(-1.1 * MM, -2.5 * MM, 2.2 * MM);
  g.add(ldo);
  for (const dy of [-2.2, 2.2]) {
    const cap = new THREE.Mesh(GEO_SMD_0603, mats.ceramicCap);
    cap.position.set(-1.0 * MM, dy * MM, -0.5 * MM);
    g.add(cap);
  }

  for (let i = 0; i < 6; ++i) {
    const pin = new THREE.Mesh(GEO_PIN_GOLD, mats.gold);
    pin.rotation.y = Math.PI / 2;
    pin.position.set(-3.2 * MM, (-6.35 + i * 2.54) * MM, 3.8 * MM);
    g.add(pin);
  }

  return g;
}

/**
 * PCA9685 16-Channel PWM Driver with Copper-Traced & Silkscreened PCB, TSSOP-28, & 48 Gold Pins.
 */
export function createPCA9685Assembly(mats) {
  const g = new THREE.Group();

  const pcb = new THREE.Mesh(new THREE.BoxGeometry(1.6 * MM, 62.0 * MM, 25.0 * MM), mats.pcbPCA9685);
  pcb.castShadow = true;
  g.add(pcb);

  // 4x Corner Knurled Brass M2 Standoffs + Threaded Stainless Bolts
  for (const sy of [-28.5, 28.5]) {
    for (const sz of [-10.0, 10.0]) {
      const standoff = new THREE.Mesh(new THREE.CylinderGeometry(2.0 * MM, 2.0 * MM, 3.2 * MM, 6), mats.brass);
      standoff.rotation.z = Math.PI / 2;
      standoff.position.set(-2.4 * MM, sy * MM, sz * MM);
      g.add(standoff);

      const bolt = createM2PanScrew(mats, 4.8);
      bolt.rotation.y = Math.PI / 2;
      bolt.position.set(0.9 * MM, sy * MM, sz * MM);
      g.add(bolt);
    }
  }

  // NXP PCA9685PW TSSOP-28 IC with individual pin comb visual
  const icBody = new THREE.Mesh(new THREE.BoxGeometry(1.2 * MM, 9.7 * MM, 4.4 * MM), mats.icLaser);
  icBody.position.set(1.3 * MM, 0, 2.2 * MM);
  g.add(icBody);
  for (let p = 0; p < 14; ++p) {
    const lead = new THREE.Mesh(new THREE.BoxGeometry(0.45 * MM, 0.32 * MM, 6.4 * MM), mats.stainless);
    lead.position.set(1.0 * MM, (-4.22 + p * 0.65) * MM, 2.2 * MM);
    g.add(lead);
  }

  const rowMats = [mats.hdrYellow, mats.hdrRed, mats.hdrBlack];
  const rowZ = [-3.6, -6.14, -8.68];
  let chIndex = 0;
  for (let bank = 0; bank < 4; ++bank) {
    const bankY0 = -24.5 + bank * 14.0;
    for (let r = 0; r < 3; ++r) {
      const strip = new THREE.Mesh(new THREE.BoxGeometry(2.5 * MM, 10.2 * MM, 2.4 * MM), rowMats[r]);
      strip.position.set(2.0 * MM, (bankY0 + 3.81) * MM, rowZ[r] * MM);
      g.add(strip);
    }
    for (let c = 0; c < 4; ++c) {
      const py = (bankY0 + c * 2.54) * MM;
      for (let r = 0; r < 3; ++r) {
        const pin = new THREE.Mesh(GEO_PIN_GOLD, mats.gold);
        pin.rotation.y = Math.PI / 2;
        pin.position.set(3.8 * MM, py, rowZ[r] * MM);
        g.add(pin);
      }
      if (chIndex < 12) {
        const plug = new THREE.Mesh(new THREE.BoxGeometry(5.6 * MM, 2.35 * MM, 7.5 * MM), mats.hdrBlack);
        plug.position.set(6.2 * MM, py, -6.14 * MM);
        g.add(plug);
      }
      chIndex++;
    }
  }

  const termBlock = new THREE.Mesh(new THREE.BoxGeometry(7.5 * MM, 10.2 * MM, 8.0 * MM), mats.termGreen);
  termBlock.position.set(4.5 * MM, 0, 8.2 * MM);
  g.add(termBlock);
  for (const sy of [-2.54, 2.54]) {
    const screw = new THREE.Mesh(new THREE.CylinderGeometry(1.7 * MM, 1.7 * MM, 1.2 * MM, 14), mats.stainless);
    screw.position.set(5.2 * MM, sy * MM, 12.1 * MM);
    g.add(screw);
  }

  const cap = new THREE.Mesh(new THREE.CylinderGeometry(3.2 * MM, 3.2 * MM, 7.0 * MM, 20), mats.capSleeve);
  cap.rotation.z = Math.PI / 2;
  cap.position.set(4.3 * MM, -14.5 * MM, 7.8 * MM);
  g.add(cap);
  const capTop = new THREE.Mesh(new THREE.CylinderGeometry(3.0 * MM, 3.0 * MM, 0.3 * MM, 20), mats.aluminum);
  capTop.rotation.z = Math.PI / 2;
  capTop.position.set(7.85 * MM, -14.5 * MM, 7.8 * MM);
  g.add(capTop);

  for (let i = 0; i < 6; ++i) {
    const i2cPin = new THREE.Mesh(GEO_PIN_GOLD, mats.gold);
    i2cPin.rotation.z = Math.PI / 2;
    i2cPin.position.set(1.8 * MM, 30.2 * MM, (-6.35 + i * 2.54) * MM);
    g.add(i2cPin);
  }

  return g;
}

/**
 * MPU-6050 (GY-521) 6-Axis MEMS IMU at CoM with Silkscreened X/Y Reference Arrows & Brass Inserts.
 */
export function createMPU6050CoMAssembly(mats) {
  const g = new THREE.Group();

  const pcb = new THREE.Mesh(new THREE.BoxGeometry(15.8 * MM, 20.5 * MM, 1.6 * MM), mats.pcbMPU6050);
  g.add(pcb);

  const qfn = new THREE.Mesh(new THREE.BoxGeometry(4.0 * MM, 4.0 * MM, 0.95 * MM), mats.icLaser);
  qfn.position.set(0, 0, 1.25 * MM);
  g.add(qfn);
  const qfnPads = new THREE.Mesh(new THREE.BoxGeometry(4.6 * MM, 4.6 * MM, 0.3 * MM), mats.gold);
  qfnPads.position.set(0, 0, 0.92 * MM);
  g.add(qfnPads);

  const ldo = new THREE.Mesh(new THREE.BoxGeometry(2.9 * MM, 1.6 * MM, 1.0 * MM), mats.icEpoxy);
  ldo.position.set(-4.2 * MM, 3.8 * MM, 1.2 * MM);
  g.add(ldo);
  for (const sy of [-4.5, -2.0, 5.5]) {
    const smd = new THREE.Mesh(GEO_SMD_0603, mats.ceramicCap);
    smd.position.set(3.8 * MM, sy * MM, 1.1 * MM);
    g.add(smd);
  }

  for (const sy of [-8.0, 8.0]) {
    const insert = new THREE.Mesh(new THREE.CylinderGeometry(1.6 * MM, 1.6 * MM, 2.2 * MM, 12), mats.brass);
    insert.rotation.x = Math.PI / 2;
    insert.position.set(0, sy * MM, -1.9 * MM);
    g.add(insert);

    const screw = createM2PanScrew(mats, 5.2);
    screw.position.set(0, sy * MM, 0.85 * MM);
    g.add(screw);
  }

  for (let i = 0; i < 4; ++i) {
    const pin = new THREE.Mesh(GEO_PIN_GOLD, mats.gold);
    pin.position.set(-6.4 * MM, (-3.81 + i * 2.54) * MM, 2.4 * MM);
    g.add(pin);
  }

  return g;
}

/**
 * Dual-Rail Isolated Power System with Copper-Poured PCBs, "4R7" Laser-Marked Inductors,
 * Printed 1000uF 10V Low-ESR Sleeve, 1N5819 Schottky, and VBAT Divider.
 */
export function createDualBuckPowerNodes(mats) {
  const buck5A = new THREE.Group();
  const pcb5A = new THREE.Mesh(new THREE.BoxGeometry(18.0 * MM, 2.0 * MM, 30.0 * MM), mats.pcbBuck);
  buck5A.add(pcb5A);

  const ind5A = new THREE.Mesh(new THREE.BoxGeometry(7.0 * MM, 4.2 * MM, 7.0 * MM), mats.ferrite4R7);
  ind5A.position.set(0.5 * MM, 2.8 * MM, -2.0 * MM);
  buck5A.add(ind5A);

  const ic5A = new THREE.Mesh(new THREE.BoxGeometry(4.0 * MM, 1.1 * MM, 4.0 * MM), mats.icLaser);
  ic5A.position.set(-3.5 * MM, 1.5 * MM, 6.5 * MM);
  buck5A.add(ic5A);

  for (const sz of [-10.5, -8.2, 8.5, 10.8]) {
    const mlcc = new THREE.Mesh(GEO_SMD_1206, mats.ceramicCap);
    mlcc.position.set(3.2 * MM, 1.6 * MM, sz * MM);
    buck5A.add(mlcc);
  }
  const pot5A = new THREE.Mesh(new THREE.CylinderGeometry(1.6 * MM, 1.6 * MM, 1.5 * MM, 14), mats.brass);
  pot5A.position.set(-4.8 * MM, 1.6 * MM, -8.5 * MM);
  buck5A.add(pot5A);

  // 1000 uF 10V Low-ESR Capacitor with Printed Sleeve + K-Scored Vent Cap + Rubber Bung
  const cap1000 = new THREE.Group();
  const capCan = new THREE.Mesh(new THREE.CylinderGeometry(4.2 * MM, 4.2 * MM, 12.0 * MM, 28), mats.capSleeve);
  capCan.rotation.x = Math.PI / 2;
  cap1000.add(capCan);

  const capVent = new THREE.Mesh(new THREE.CylinderGeometry(3.95 * MM, 3.95 * MM, 0.35 * MM, 24), mats.aluminum);
  capVent.rotation.x = Math.PI / 2;
  capVent.position.z = 6.05 * MM;
  cap1000.add(capVent);

  // K-score safety vent lines on capacitor top
  for (const ang of [0, Math.PI * 0.5]) {
    const score = new THREE.Mesh(new THREE.BoxGeometry(6.8 * MM, 0.25 * MM, 0.18 * MM), mats.darkSteel);
    score.rotation.z = ang;
    score.position.z = 6.22 * MM;
    cap1000.add(score);
  }

  const bung = new THREE.Mesh(new THREE.CylinderGeometry(3.9 * MM, 3.9 * MM, 0.8 * MM, 20), mats.soleSilicone);
  bung.rotation.x = Math.PI / 2;
  bung.position.z = -5.8 * MM;
  cap1000.add(bung);

  for (const dx of [-1.75, 1.75]) {
    const lead = new THREE.Mesh(new THREE.CylinderGeometry(0.35 * MM, 0.35 * MM, 5.2 * MM, 8), mats.stainless);
    lead.rotation.x = Math.PI / 2;
    lead.position.set(dx * MM, 0, -8.2 * MM);
    cap1000.add(lead);
  }

  // Mini-360 5.0V 2A Logic Buck + 1N5819 Schottky + VBAT Divider
  const buck2A = new THREE.Group();
  const pcb2A = new THREE.Mesh(new THREE.BoxGeometry(11.2 * MM, 1.8 * MM, 17.5 * MM), mats.pcbBuck);
  buck2A.add(pcb2A);

  const ind2A = new THREE.Mesh(new THREE.BoxGeometry(4.4 * MM, 3.0 * MM, 4.4 * MM), mats.ferrite4R7);
  ind2A.position.set(0, -2.2 * MM, -2.5 * MM);
  buck2A.add(ind2A);

  const ic2A = new THREE.Mesh(new THREE.BoxGeometry(4.9 * MM, 1.4 * MM, 3.9 * MM), mats.icLaser);
  ic2A.position.set(0.5 * MM, -1.5 * MM, 4.0 * MM);
  buck2A.add(ic2A);

  const pot2A = new THREE.Mesh(new THREE.CylinderGeometry(1.8 * MM, 1.8 * MM, 1.4 * MM, 14), mats.brass);
  pot2A.position.set(-2.5 * MM, -1.5 * MM, -5.8 * MM);
  buck2A.add(pot2A);

  const schottkyBody = new THREE.Mesh(new THREE.CylinderGeometry(1.35 * MM, 1.35 * MM, 4.8 * MM, 16), mats.icEpoxy);
  schottkyBody.position.set(0, -2.6 * MM, 11.8 * MM);
  schottkyBody.rotation.x = Math.PI / 2;
  buck2A.add(schottkyBody);
  const cathodeBand = new THREE.Mesh(new THREE.CylinderGeometry(1.40 * MM, 1.40 * MM, 0.95 * MM, 16), mats.aluminum);
  cathodeBand.rotation.x = Math.PI / 2;
  cathodeBand.position.set(0, -2.6 * MM, 13.3 * MM);
  buck2A.add(cathodeBand);
  const axialLead = new THREE.Mesh(new THREE.CylinderGeometry(0.38 * MM, 0.38 * MM, 8.8 * MM, 8), mats.stainless);
  axialLead.rotation.x = Math.PI / 2;
  axialLead.position.set(0, -2.6 * MM, 11.8 * MM);
  buck2A.add(axialLead);

  for (const dz of [9.5, 11.2]) {
    const rDiv = new THREE.Mesh(GEO_SMD_0603, mats.icEpoxy);
    rDiv.position.set(3.6 * MM, -1.4 * MM, dz * MM);
    buck2A.add(rDiv);
  }

  return { buck5A, cap1000, buck2A };
}

/**
 * MAX98357A I2S 3W Class-D Amp + 28 mm 4ohm 3W Speaker WITH INTERNAL COPPER VOICE COIL & NOMEX SPIDER!
 */
export function createAudioSystemNodes(mats) {
  const ampGroup = new THREE.Group();
  const pcb = new THREE.Mesh(new THREE.BoxGeometry(17.8 * MM, 1.6 * MM, 19.0 * MM), mats.pcbPurple);
  ampGroup.add(pcb);

  const qfn = new THREE.Mesh(new THREE.BoxGeometry(3.2 * MM, 1.0 * MM, 3.2 * MM), mats.icLaser);
  qfn.position.set(0, -1.2 * MM, 0);
  ampGroup.add(qfn);

  for (const dx of [-3.5, 3.5]) {
    const fb = new THREE.Mesh(GEO_SMD_1206, mats.ferrite4R7);
    fb.position.set(dx * MM, -1.3 * MM, -2.8 * MM);
    ampGroup.add(fb);
  }

  const spkTerm = new THREE.Mesh(new THREE.BoxGeometry(6.5 * MM, 6.2 * MM, 5.2 * MM), mats.termGreen);
  spkTerm.position.set(3.5 * MM, -3.6 * MM, -6.2 * MM);
  ampGroup.add(spkTerm);

  for (let i = 0; i < 7; ++i) {
    const pin = new THREE.Mesh(GEO_PIN_GOLD, mats.gold);
    pin.position.set((-6.5 + i * 2.1) * MM, -2.2 * MM, 7.5 * MM);
    ampGroup.add(pin);
  }

  // 28 mm Neodymium Speaker with Internal Voice Coil, Kapton Bobbin, Nomex Spider & T-Yoke!
  const spkGroup = new THREE.Group();
  const rim = new THREE.Mesh(new THREE.CylinderGeometry(14.0 * MM, 12.8 * MM, 2.2 * MM, 32), mats.anodizedBlack);
  rim.rotation.z = Math.PI / 2;
  rim.position.x = 1.2 * MM;
  spkGroup.add(rim);

  const surround = new THREE.Mesh(new THREE.TorusGeometry(11.4 * MM, 1.35 * MM, 12, 32), mats.soleSilicone);
  surround.rotation.y = Math.PI / 2;
  surround.position.x = 2.0 * MM;
  spkGroup.add(surround);

  const cone = new THREE.Mesh(new THREE.ConeGeometry(10.6 * MM, 3.4 * MM, 32, 1, true), mats.speakerCone);
  cone.rotation.z = Math.PI / 2;
  cone.position.x = 0.5 * MM;
  spkGroup.add(cone);

  const dustCap = new THREE.Mesh(
    new THREE.SphereGeometry(4.2 * MM, 20, 12, 0, Math.PI * 2, 0, Math.PI * 0.45),
    mats.anodizedBlack
  );
  dustCap.rotation.z = -Math.PI / 2;
  dustCap.position.x = -0.2 * MM;
  spkGroup.add(dustCap);

  // Internal Corrugated Nomex Spider Suspension Ring + Wound Copper Voice Coil!
  const spider = new THREE.Mesh(new THREE.CylinderGeometry(8.8 * MM, 8.8 * MM, 0.35 * MM, 24), mats.nomexSpider);
  spider.rotation.z = Math.PI / 2;
  spider.position.x = -1.3 * MM;
  spkGroup.add(spider);

  const voiceCoil = new THREE.Mesh(new THREE.CylinderGeometry(4.3 * MM, 4.3 * MM, 3.6 * MM, 24), mats.copper);
  voiceCoil.rotation.z = Math.PI / 2;
  voiceCoil.position.x = -2.2 * MM;
  spkGroup.add(voiceCoil);

  const basket = new THREE.Mesh(new THREE.CylinderGeometry(11.5 * MM, 8.2 * MM, 2.6 * MM, 24, 1, true), mats.darkSteel);
  basket.rotation.z = Math.PI / 2;
  basket.position.x = -1.2 * MM;
  spkGroup.add(basket);

  const neoMag = new THREE.Mesh(new THREE.CylinderGeometry(7.8 * MM, 7.8 * MM, 3.4 * MM, 24), mats.aluminum);
  neoMag.rotation.z = Math.PI / 2;
  neoMag.position.x = -4.1 * MM;
  spkGroup.add(neoMag);

  for (const sy of [-5.5, 5.5]) {
    const lug = new THREE.Mesh(new THREE.BoxGeometry(1.8 * MM, 2.2 * MM, 0.5 * MM), mats.gold);
    lug.position.set(-2.6 * MM, sy * MM, -8.2 * MM);
    spkGroup.add(lug);
  }

  return { ampGroup, spkGroup };
}

/**
 * 2S 14500 800mAh 10C Li-ion Pack with Printed Cell Wraps, Helical Nickel Springs,
 * Silkscreened 2S 5A BMS, XH2.54 Balance Sense Wires, USB-C Charger & 6A Slide Switch!
 */
export function createBattery2SAndRearIO(mats) {
  const packGroup = new THREE.Group();

  const cradle = new THREE.Mesh(new THREE.BoxGeometry(8.0 * MM, 31.5 * MM, 52.4 * MM), mats.anodizedBlack);
  cradle.position.set(3.2 * MM, 0, 0);
  packGroup.add(cradle);

  const cells = [];
  for (const sy of [-8.0, 8.0]) {
    const cellGrp = new THREE.Group();
    cellGrp.position.set(0, sy * MM, 0);

    const wrap = new THREE.Mesh(new THREE.CylinderGeometry(7.0 * MM, 7.0 * MM, 49.0 * MM, 32), mats.liIonBlue);
    wrap.rotation.x = Math.PI / 2;
    wrap.rotation.y = Math.PI * 0.5;
    wrap.castShadow = true;
    cellGrp.add(wrap);

    const polDir = sy < 0 ? 1 : -1;
    const gasket = new THREE.Mesh(new THREE.CylinderGeometry(6.6 * MM, 6.6 * MM, 0.6 * MM, 20), mats.nylonWhite);
    gasket.rotation.x = Math.PI / 2;
    gasket.position.z = polDir * 24.8 * MM;
    cellGrp.add(gasket);

    const posCap = new THREE.Mesh(new THREE.CylinderGeometry(3.8 * MM, 4.2 * MM, 1.4 * MM, 18), mats.stainless);
    posCap.rotation.x = Math.PI / 2;
    posCap.position.z = polDir * 25.4 * MM;
    cellGrp.add(posCap);

    // Conical Helical Nickel Negative Contact Spring inside Cradle!
    for (let s = 0; s < 3; ++s) {
      const coil = new THREE.Mesh(new THREE.TorusGeometry((3.8 - s * 0.7) * MM, 0.32 * MM, 6, 16), mats.stainless);
      coil.position.set(0, sy * MM, -polDir * (25.2 + s * 0.45) * MM);
      packGroup.add(coil);
    }

    packGroup.add(cellGrp);
    cells.push(cellGrp);
  }

  // Spot-Welded Pure Nickel Series Busbar Strip + 4 weld dimples
  const nickelSeries = new THREE.Mesh(new THREE.BoxGeometry(4.8 * MM, 21.2 * MM, 0.45 * MM), mats.stainless);
  nickelSeries.position.set(0, 0, 25.9 * MM);
  packGroup.add(nickelSeries);

  // Silkscreened 2S 5A BMS PCB + 4x 8205A Dual MOSFETs + R010 Current-Sense Shunt
  const bmsGroup = new THREE.Group();
  bmsGroup.position.set(-8.2 * MM, 0, 2.0 * MM);
  const bmsPcb = new THREE.Mesh(new THREE.BoxGeometry(1.8 * MM, 28.0 * MM, 10.5 * MM), mats.pcbBMS);
  bmsGroup.add(bmsPcb);
  for (const dy of [-9.0, -3.5, 2.0, 7.5]) {
    const mosfet = new THREE.Mesh(new THREE.BoxGeometry(1.1 * MM, 3.2 * MM, 4.4 * MM), mats.icLaser);
    mosfet.position.set(-1.3 * MM, dy * MM, 0);
    bmsGroup.add(mosfet);
  }
  const shunt = new THREE.Mesh(GEO_SMD_1206, mats.icEpoxy);
  shunt.position.set(-1.2 * MM, 11.2 * MM, -2.0 * MM);
  bmsGroup.add(shunt);
  packGroup.add(bmsGroup);

  // Rear Service Panel IO (USB-C 2S Boost Charger & 6A Main Slide Switch)
  const rearIOGroup = new THREE.Group();
  const usbcPcb = new THREE.Mesh(new THREE.BoxGeometry(6.5 * MM, 11.0 * MM, 2.0 * MM), mats.pcbXIAO);
  usbcPcb.position.set(-19.8 * MM, -14.0 * MM, 12.6 * MM);
  rearIOGroup.add(usbcPcb);
  const usbcPort = new THREE.Mesh(new THREE.BoxGeometry(7.2 * MM, 8.94 * MM, 3.2 * MM), mats.stainless);
  usbcPort.position.set(-22.2 * MM, -14.0 * MM, 14.5 * MM);
  rearIOGroup.add(usbcPort);
  const usbcCav = new THREE.Mesh(new THREE.BoxGeometry(1.2 * MM, 7.2 * MM, 1.8 * MM), mats.icEpoxy);
  usbcCav.position.set(-25.4 * MM, -14.0 * MM, 14.5 * MM);
  rearIOGroup.add(usbcCav);

  const swBody = new THREE.Mesh(new THREE.BoxGeometry(5.5 * MM, 11.2 * MM, 5.0 * MM), mats.stainless);
  swBody.position.set(-21.2 * MM, 14.0 * MM, 14.5 * MM);
  rearIOGroup.add(swBody);
  const swKnob = new THREE.Mesh(new THREE.BoxGeometry(3.4 * MM, 4.2 * MM, 3.4 * MM), mats.anodizedBlack);
  swKnob.position.set(-25.0 * MM, 15.6 * MM, 14.5 * MM);
  rearIOGroup.add(swKnob);
  for (const dy of [-3.5, 0, 3.5]) {
    const lug = new THREE.Mesh(new THREE.BoxGeometry(3.2 * MM, 0.8 * MM, 1.4 * MM), mats.stainless);
    lug.position.set(-17.2 * MM, (14.0 + dy) * MM, 14.5 * MM);
    rearIOGroup.add(lug);
  }

  return { packGroup, cells, bmsGroup, rearIOGroup };
}

/**
 * Molded 2.0 mm High-Friction Silicone Sole Pad (mu >= 0.90) with Directional Chevron Tread Ribs!
 */
export function createSiliconeTreadPad(mats) {
  const g = new THREE.Group();
  const base = new THREE.Mesh(new THREE.BoxGeometry(43.6 * MM, 39.6 * MM, 1.5 * MM), mats.soleSilicone);
  base.position.z = 0.25 * MM;
  base.receiveShadow = true;
  g.add(base);

  // 7 directional anti-slip tread ribs along bottom contact face
  for (let ix = -3; ix <= 3; ++ix) {
    const rib = new THREE.Mesh(new THREE.BoxGeometry(3.4 * MM, 36.5 * MM, 0.65 * MM), mats.soleSilicone);
    rib.position.set(ix * 6.0 * MM, 0, -0.68 * MM);
    g.add(rib);
  }
  return g;
}

export function createWireTube(ptsMM, radiusMM, mat, tubularSegments = 28) {
  const vecs = ptsMM.map((p) => new THREE.Vector3(p[0] * MM, p[1] * MM, p[2] * MM));
  const curve = new THREE.CatmullRomCurve3(vecs);
  const geo = new THREE.TubeGeometry(curve, tubularSegments, radiusMM * MM, 8, false);
  return new THREE.Mesh(geo, mat);
}

/**
 * Complete 12-Channel + Dual-Buck + 2S Balance + VBAT_ADC + OE# + I2C + I2S Wiring Harness!
 */
export function createCompleteWiringHarnesses(mats) {
  const torsoHarness = new THREE.Group();

  // 1. VBAT 7.4V 22AWG + 2S Balance Sense (BM 3.7V)
  torsoHarness.add(createWireTube([
    [-18.0, 2.0, 32.0], [-17.5, 11.0, 21.0], [-17.0, 14.0, 14.5],
    [-8.0, 16.5, 16.0], [6.0, 20.0, 20.0], [11.0, 21.0, 23.0]
  ], 0.85, mats.wireVbatRed));
  torsoHarness.add(createWireTube([
    [-17.0, 14.0, 14.5], [-9.0, 2.0, 15.5], [4.0, -14.0, 18.0], [11.0, -21.0, 22.0]
  ], 0.90, mats.wireVbatRed));
  torsoHarness.add(createWireTube([
    [-18.0, -2.0, 30.0], [-11.0, -10.0, 20.0], [4.0, -18.0, 21.0], [11.0, -21.0, 25.0]
  ], 0.85, mats.wireGndBlack));
  // 2S Center-Tap BM 3.7V Balance Wire from Top Nickel Strip -> BMS BM Pad
  torsoHarness.add(createWireTube([
    [-10.5, 0.0, 57.8], [-16.8, 0.0, 52.0], [-18.5, 0.0, 35.0]
  ], 0.52, mats.wirePwmSig, 18));

  // 2. 5.6V 5A Servo High-Current Rail -> 1000uF Cap -> PCA9685 V+ Screw Terminal
  torsoHarness.add(createWireTube([
    [11.0, -20.0, 28.0], [13.8, -14.5, 19.5], [10.5, -8.0, 32.0], [5.8, -2.5, 41.5]
  ], 0.95, mats.wireServo5v6));
  torsoHarness.add(createWireTube([
    [11.0, -20.0, 26.0], [13.2, -13.0, 18.5], [9.5, -5.0, 32.0], [5.8, 2.5, 41.5]
  ], 0.90, mats.wireGndBlack));

  // 3. 5.0V Isolated Logic + VBAT_ADC (GPIO7) + OE# (GPIO8) Control Leads
  torsoHarness.add(createWireTube([
    [11.0, 20.0, 34.5], [12.0, 20.0, 42.0], [7.5, 11.0, 52.0], [-2.0, 2.0, 61.0]
  ], 0.72, mats.wireLogic5v0));
  torsoHarness.add(createWireTube([
    [13.0, 19.5, 32.5], [6.5, 25.0, 38.0], [3.0, 29.5, 36.5], [-1.8, 1.0, 61.0]
  ], 0.50, mats.wireSenseTeal, 22));

  // 4. 400 kHz I2C Bus (SDA=Blue, SCL=Amber)
  torsoHarness.add(createWireTube([
    [-2.5, -1.5, 61.0], [2.0, 18.0, 48.0], [3.0, 29.5, 34.0], [3.5, 14.0, 19.0], [-4.2, -1.2, 15.5]
  ], 0.62, mats.wireI2cSda));
  torsoHarness.add(createWireTube([
    [-2.5, -2.8, 61.0], [1.2, 17.0, 47.2], [3.0, 29.5, 31.5], [2.8, 13.0, 18.2], [-4.2, 1.2, 15.5]
  ], 0.62, mats.wireI2cScl));

  // 5. I2S Audio + Speaker Twisted Leads
  torsoHarness.add(createWireTube([
    [-1.5, 3.0, 60.5], [6.5, 14.0, 51.0], [11.5, 19.0, 44.5]
  ], 0.68, mats.wireI2sAudio));
  torsoHarness.add(createWireTube([
    [14.5, 18.0, 34.0], [17.2, 10.5, 29.0], [17.8, 5.5, 26.0]
  ], 0.62, mats.wireVbatRed));
  torsoHarness.add(createWireTube([
    [14.5, 16.5, 33.2], [17.0, 8.5, 28.0], [17.8, -5.5, 26.0]
  ], 0.62, mats.wireGndBlack));

  // 6. Individual Shoulder (CH8/CH9) & Pelvis Hip (CH0..CH7) Servo Ribbon Bundles from PCA9685!
  for (const side of [-1, 1]) {
    // Shoulder servo lead
    torsoHarness.add(createWireTube([
      [7.2, side * 12.0, 27.0], [6.5, side * 24.0, 36.0], [-8.0, side * 28.0, 44.0]
    ], 0.56, mats.wirePwmSig, 18));
    // Pelvis Hip Roll & Hip Pitch trunk down into hip clevis
    torsoHarness.add(createWireTube([
      [7.2, side * 19.0, 26.0], [6.0, side * 22.0, 14.0], [-5.5, side * 23.0, 2.0]
    ], 0.64, mats.wireServo5v6, 18));
  }

  // 7. Head Internal Coaxial Harness
  const headHarness = new THREE.Group();
  headHarness.add(createWireTube([
    [8.0, -7.5, 12.0], [10.5, -6.0, 8.5], [15.5, -3.5, 7.8]
  ], 0.55, mats.wireI2cSda, 16));
  headHarness.add(createWireTube([
    [8.0, -7.5, 10.2], [10.5, -4.5, 7.5], [15.5, -1.0, 7.8]
  ], 0.55, mats.wireI2cScl, 16));
  headHarness.add(createWireTube([
    [8.0, 7.5, 12.0], [4.0, 3.5, 4.0], [-1.5, 0.0, -2.0]
  ], 0.68, mats.wireLogic5v0, 16));

  // 8. Left & Right Leg Internal Conduit Harnesses (inside leg_link.scad's X=-6.8mm rear bore!)
  function makeLegHarness() {
    const lg = new THREE.Group();
    lg.add(createWireTube([
      [-3.5, 0.0, -4.0], [-6.8, -1.1, -20.0], [-6.8, -1.1, -58.0], [-5.5, -0.9, -79.0]
    ], 0.65, mats.wireServo5v6, 20));
    lg.add(createWireTube([
      [-3.5, 1.1, -4.0], [-6.8, 0.0, -20.0], [-6.8, 0.0, -58.0], [-5.2, 0.2, -85.0]
    ], 0.60, mats.wirePwmSig, 20));
    lg.add(createWireTube([
      [-3.5, -1.1, -4.0], [-6.8, 1.1, -20.0], [-6.8, 1.1, -58.0], [-5.5, 0.9, -81.0]
    ], 0.60, mats.wireGndBlack, 20));
    return lg;
  }

  return {
    torsoHarness,
    headHarness,
    leftLegHarness: makeLegHarness(),
    rightLegHarness: makeLegHarness(),
  };
}

/**
 * Kinematic 12-DOF Joint Axis Vectors + MuJoCo-Verified Whole-Body Center of Mass (CoM) Target.
 */
export function attachKinematicAndCoMGuides({
  torsoGroup, headYawJoint, headPitchJoint, lShoulderJoint, rShoulderJoint, leftLeg, rightLeg
}) {
  const markers = [];
  const matAxisX = new THREE.LineBasicMaterial({ color: 0xef4444, transparent: true, opacity: 0.80 });
  const matAxisY = new THREE.LineBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.80 });
  const matAxisZ = new THREE.LineBasicMaterial({ color: 0x2563eb, transparent: true, opacity: 0.80 });
  const matCoM   = new THREE.MeshStandardMaterial({ color: 0xf59e0b, roughness: 0.20, metalness: 0.85, emissive: 0xb45309, emissiveIntensity: 0.25 });

  function addAxisLine(parent, p1, p2, mat) {
    const g = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(p1[0] * MM, p1[1] * MM, p1[2] * MM),
      new THREE.Vector3(p2[0] * MM, p2[1] * MM, p2[2] * MM),
    ]);
    const line = new THREE.Line(g, mat);
    line.visible = false;
    parent.add(line);
    markers.push(line);
  }

  addAxisLine(headYawJoint,   [0, 0, -10], [0, 0, 26], matAxisZ);
  addAxisLine(headPitchJoint, [0, -24, 0], [0, 24, 0], matAxisY);
  addAxisLine(lShoulderJoint, [0, -14, 0], [0, 14, 0], matAxisY);
  addAxisLine(rShoulderJoint, [0, -14, 0], [0, 14, 0], matAxisY);
  for (const leg of [leftLeg, rightLeg]) {
    addAxisLine(leg.hipRoll,    [-20, 0, 0], [20, 0, 0], matAxisX);
    addAxisLine(leg.hipPitch,   [0, -20, 0], [0, 20, 0], matAxisY);
    addAxisLine(leg.anklePitch, [0, -20, 0], [0, 20, 0], matAxisY);
    addAxisLine(leg.ankleRoll,  [-22, 0, 0], [22, 0, 0], matAxisX);
  }

  const comGroup = new THREE.Group();
  comGroup.position.set(0.99 * MM, 0, -12.45 * MM);
  comGroup.visible = false;

  comGroup.add(new THREE.Mesh(new THREE.SphereGeometry(3.4 * MM, 20, 20), matCoM));
  comGroup.add(new THREE.Mesh(new THREE.TorusGeometry(6.0 * MM, 0.45 * MM, 10, 32), matCoM));

  const plumbGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0, 0, -111.55 * MM),
  ]);
  const plumbLine = new THREE.Line(plumbGeo, new THREE.LineDashedMaterial({
    color: 0xd97706, dashSize: 0.005, gapSize: 0.003, transparent: true, opacity: 0.78
  }));
  plumbLine.computeLineDistances();
  comGroup.add(plumbLine);

  const polyPts = [
    [-25.99, -49, -111.55], [31.01, -49, -111.55],
    [31.01,   49, -111.55], [-25.99, 49, -111.55], [-25.99, -49, -111.55],
  ].map((p) => new THREE.Vector3(p[0] * MM, p[1] * MM, p[2] * MM));
  comGroup.add(new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(polyPts),
    new THREE.LineBasicMaterial({ color: 0xd97706, transparent: true, opacity: 0.58 })
  ));

  torsoGroup.add(comGroup);
  markers.push(comGroup);

  return markers;
}
