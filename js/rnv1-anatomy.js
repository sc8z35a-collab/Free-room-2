// RNV1 Ultra-High-Definition 3D CAD Mechatronics Anatomy Builder
// Every dimension, pinout, IC package, fastener, gear stage, and wire bus is grounded directly in:
//   - bom/rnv1_bom.json & bom/wiring_pinout.json
//   - cad/rnv1_params.scad & cad/parts/*.scad
//   - simulation/rnv1_humanoid.xml (Exact CoM & 12-DOF joint frames)
import * as THREE from '../vendor/three/three.module.min.js';

export const MM = 0.001;

// --- Shared Micro-Geometries (Instanced/Reused for crisp 60 FPS rendering) ---
const GEO_PIN_GOLD     = new THREE.BoxGeometry(0.64 * MM, 0.64 * MM, 6.2 * MM);
const GEO_M2_HEAD      = new THREE.CylinderGeometry(1.9 * MM, 1.9 * MM, 1.8 * MM, 16);
const GEO_M2_HEX       = new THREE.CylinderGeometry(0.85 * MM, 0.85 * MM, 0.9 * MM, 6);
const GEO_M2_PAN       = new THREE.CylinderGeometry(1.85 * MM, 1.55 * MM, 1.2 * MM, 16);
const GEO_SMD_0603     = new THREE.BoxGeometry(1.6 * MM, 0.8 * MM, 0.7 * MM);
const GEO_SMD_1206     = new THREE.BoxGeometry(3.2 * MM, 1.6 * MM, 1.2 * MM);
const GEO_HEATSINK_PIN = new THREE.BoxGeometry(3.4 * MM, 1.25 * MM, 1.25 * MM);

/**
 * Weld duplicate STL triangle soup vertices & compute smooth creased normals
 * so rounded CAD fillets render silky-smooth while hard 30deg+ edges stay razor-sharp.
 */
export function computeCreasedNormals(stlGeo, creaseAngleDeg = 30.0) {
  const posAttr = stlGeo.attributes.position;
  if (!posAttr) return stlGeo;
  const count = posAttr.count;
  const cosThresh = Math.cos((creaseAngleDeg * Math.PI) / 180.0);

  // Compute per-face unit normals
  const faceNormals = new Float32Array(count * 3);
  const vA = new THREE.Vector3(), vB = new THREE.Vector3(), vC = new THREE.Vector3();
  const cb = new THREE.Vector3(), ab = new THREE.Vector3();

  const vertMap = new Map();
  const quant = 1e5; // 0.01 mm precision after MM scaling

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
 * Create curated physically-based industrial CAD materials.
 * Shell & outer servo housings accept dynamic section-cutaway clippingPlanes,
 * while internal gears, motors, PCBs, fasteners, and wires remain whole inside cut cavities!
 */
export function createAnatomyMaterials(sectionPlanes) {
  return {
    // Sectionable Structural & Servo Shells
    shellPrimary: new THREE.MeshStandardMaterial({
      color: 0xf4f6f9, roughness: 0.38, metalness: 0.05,
      side: THREE.DoubleSide, clippingPlanes: sectionPlanes, clipShadows: true,
    }),
    shellAccent: new THREE.MeshStandardMaterial({
      color: 0xe3e8f0, roughness: 0.44, metalness: 0.08,
      side: THREE.DoubleSide, clippingPlanes: sectionPlanes, clipShadows: true,
    }),
    visor: new THREE.MeshPhysicalMaterial({
      color: 0x121722, roughness: 0.10, metalness: 0.25, clearcoat: 1.0, clearcoatRoughness: 0.08,
      side: THREE.DoubleSide, clippingPlanes: sectionPlanes, clipShadows: true,
    }),
    servoMG90S: new THREE.MeshStandardMaterial({
      color: 0x1b4785, roughness: 0.34, metalness: 0.12,
      side: THREE.DoubleSide, clippingPlanes: sectionPlanes, clipShadows: true,
    }),
    servoSG90: new THREE.MeshPhysicalMaterial({
      color: 0x2979cf, roughness: 0.22, metalness: 0.05,
      transparent: true, opacity: 0.72, clearcoat: 0.6,
      side: THREE.DoubleSide, clippingPlanes: sectionPlanes, clipShadows: true,
    }),

    // Unclipped Internal Precision Mechatronics Materials
    soleSilicone:  new THREE.MeshStandardMaterial({ color: 0x23272e, roughness: 0.88, metalness: 0.02 }),
    stainless:     new THREE.MeshStandardMaterial({ color: 0xd8dfe8, roughness: 0.22, metalness: 0.88 }),
    darkSteel:     new THREE.MeshStandardMaterial({ color: 0x38404c, roughness: 0.32, metalness: 0.80 }),
    brass:         new THREE.MeshStandardMaterial({ color: 0xd4a647, roughness: 0.24, metalness: 0.88 }),
    gold:          new THREE.MeshStandardMaterial({ color: 0xebb83b, roughness: 0.18, metalness: 0.92 }),
    copper:        new THREE.MeshStandardMaterial({ color: 0xc96b38, roughness: 0.26, metalness: 0.86 }),
    anodizedBlack: new THREE.MeshStandardMaterial({ color: 0x1b1f26, roughness: 0.32, metalness: 0.76 }),
    aluminum:      new THREE.MeshStandardMaterial({ color: 0xcdd4de, roughness: 0.25, metalness: 0.84 }),
    nylonWhite:    new THREE.MeshStandardMaterial({ color: 0xf2f5f9, roughness: 0.36, metalness: 0.03 }),
    pcbBlack:      new THREE.MeshStandardMaterial({ color: 0x171b22, roughness: 0.44, metalness: 0.12 }),
    pcbBlue:       new THREE.MeshStandardMaterial({ color: 0x124078, roughness: 0.42, metalness: 0.12 }),
    pcbPurple:     new THREE.MeshStandardMaterial({ color: 0x4f2673, roughness: 0.42, metalness: 0.12 }),
    pcbGreen:      new THREE.MeshStandardMaterial({ color: 0x15653c, roughness: 0.44, metalness: 0.12 }),
    icEpoxy:       new THREE.MeshStandardMaterial({ color: 0x14171c, roughness: 0.48, metalness: 0.15 }),
    ferrite:       new THREE.MeshStandardMaterial({ color: 0x262a32, roughness: 0.56, metalness: 0.22 }),
    ceramicCap:    new THREE.MeshStandardMaterial({ color: 0xb58852, roughness: 0.38, metalness: 0.10 }),
    termGreen:     new THREE.MeshStandardMaterial({ color: 0x1f8b4c, roughness: 0.40, metalness: 0.05 }),
    hdrYellow:     new THREE.MeshStandardMaterial({ color: 0xeab308, roughness: 0.42, metalness: 0.04 }),
    hdrRed:        new THREE.MeshStandardMaterial({ color: 0xdc2626, roughness: 0.42, metalness: 0.04 }),
    hdrBlack:      new THREE.MeshStandardMaterial({ color: 0x181b20, roughness: 0.48, metalness: 0.04 }),
    liIonBlue:     new THREE.MeshStandardMaterial({ color: 0x1b569e, roughness: 0.28, metalness: 0.32 }),
    capSleeve:     new THREE.MeshStandardMaterial({ color: 0x172554, roughness: 0.32, metalness: 0.25 }),
    fpcGold:       new THREE.MeshStandardMaterial({ color: 0xd98a2b, roughness: 0.34, metalness: 0.45, side: THREE.DoubleSide }),
    speakerCone:   new THREE.MeshStandardMaterial({ color: 0x1e2229, roughness: 0.72, metalness: 0.06, side: THREE.DoubleSide }),
    lensGlass:     new THREE.MeshPhysicalMaterial({ color: 0x0a2540, roughness: 0.04, metalness: 0.10, clearcoat: 1.0, reflectivity: 0.95 }),
    tofWindow:     new THREE.MeshPhysicalMaterial({ color: 0x2b1222, roughness: 0.08, metalness: 0.20, clearcoat: 1.0 }),

    // Silicone Wire Insulation Palette
    wireVbatRed:   new THREE.MeshStandardMaterial({ color: 0xdc2626, roughness: 0.52, metalness: 0.04 }),
    wireGndBlack:  new THREE.MeshStandardMaterial({ color: 0x1f242d, roughness: 0.55, metalness: 0.04 }),
    wireServo5v6:  new THREE.MeshStandardMaterial({ color: 0xea580c, roughness: 0.50, metalness: 0.04 }),
    wireLogic5v0:  new THREE.MeshStandardMaterial({ color: 0x10b981, roughness: 0.50, metalness: 0.04 }),
    wireI2cSda:    new THREE.MeshStandardMaterial({ color: 0x0284c7, roughness: 0.48, metalness: 0.04 }),
    wireI2cScl:    new THREE.MeshStandardMaterial({ color: 0xd97706, roughness: 0.48, metalness: 0.04 }),
    wireI2sAudio:  new THREE.MeshStandardMaterial({ color: 0x7c3aed, roughness: 0.48, metalness: 0.04 }),
    wirePwmSig:    new THREE.MeshStandardMaterial({ color: 0xf59e0b, roughness: 0.50, metalness: 0.04 }),
  };
}

/**
 * DIN 912 M2 Stainless Socket-Head Cap Screw (head aligned at Z=0..+1.8mm, shaft along -Z).
 */
export function createM2SocketBolt(mats, shaftLenMM = 6.0) {
  const g = new THREE.Group();
  const head = new THREE.Mesh(GEO_M2_HEAD, mats.stainless);
  head.rotation.x = Math.PI / 2;
  head.position.z = 0.9 * MM;
  g.add(head);

  const hex = new THREE.Mesh(GEO_M2_HEX, mats.darkSteel);
  hex.rotation.x = Math.PI / 2;
  hex.position.z = 1.45 * MM;
  g.add(hex);

  const shaft = new THREE.Mesh(
    new THREE.CylinderGeometry(0.98 * MM, 0.95 * MM, shaftLenMM * MM, 12),
    mats.stainless
  );
  shaft.rotation.x = Math.PI / 2;
  shaft.position.z = -(shaftLenMM * 0.5) * MM;
  g.add(shaft);
  return g;
}

/**
 * M2 Pan-Head Machine/Self-Tapping Screw (head at Z=0..+1.2mm, shaft along -Z).
 */
export function createM2PanScrew(mats, shaftLenMM = 5.0) {
  const g = new THREE.Group();
  const head = new THREE.Mesh(GEO_M2_PAN, mats.stainless);
  head.rotation.x = Math.PI / 2;
  head.position.z = 0.6 * MM;
  g.add(head);

  const slot1 = new THREE.Mesh(new THREE.BoxGeometry(2.1 * MM, 0.42 * MM, 0.35 * MM), mats.darkSteel);
  slot1.position.z = 1.15 * MM;
  g.add(slot1);
  const slot2 = new THREE.Mesh(new THREE.BoxGeometry(0.42 * MM, 2.1 * MM, 0.35 * MM), mats.darkSteel);
  slot2.position.z = 1.15 * MM;
  g.add(slot2);

  const shaft = new THREE.Mesh(
    new THREE.CylinderGeometry(0.95 * MM, 0.85 * MM, shaftLenMM * MM, 10),
    mats.stainless
  );
  shaft.rotation.x = Math.PI / 2;
  shaft.position.z = -(shaftLenMM * 0.5) * MM;
  g.add(shaft);
  return g;
}

/**
 * Opposing Coaxial Ø4.0 mm Dual-Shear Flanged Brass Sleeve Bushing + Retaining Washer.
 * Local origin at joint bore face, axis along +Z.
 */
export function createDualShearBushing(mats) {
  const g = new THREE.Group();
  const sleeve = new THREE.Mesh(new THREE.CylinderGeometry(2.0 * MM, 2.0 * MM, 3.8 * MM, 18), mats.brass);
  sleeve.rotation.x = Math.PI / 2;
  sleeve.position.z = -1.9 * MM;
  g.add(sleeve);

  const flange = new THREE.Mesh(new THREE.CylinderGeometry(3.3 * MM, 3.3 * MM, 0.7 * MM, 18), mats.brass);
  flange.rotation.x = Math.PI / 2;
  flange.position.z = 0.35 * MM;
  g.add(flange);

  const pin = new THREE.Mesh(new THREE.CylinderGeometry(1.35 * MM, 1.35 * MM, 4.6 * MM, 14), mats.stainless);
  pin.rotation.x = Math.PI / 2;
  pin.position.z = -1.6 * MM;
  g.add(pin);
  return g;
}

/**
 * Molded Servo Horn + Central M2x5 Retaining Screw + Radial Self-Tapping Screws.
 * Attached to the CHILD moving link so it rotates coaxially with the limb!
 * Local origin (0,0,0) = joint center, spline bore along -Z, horn arm along +X.
 */
export function createServoHornAssembly(mats, { isSG90 = false, armLenMM = 12.0, doubleArm = false } = {}) {
  const g = new THREE.Group();
  const hornMat = isSG90 ? mats.nylonWhite : mats.anodizedBlack;

  // Spline hub barrel
  const hub = new THREE.Mesh(new THREE.CylinderGeometry(3.5 * MM, 3.5 * MM, 3.2 * MM, 18), hornMat);
  hub.rotation.x = Math.PI / 2;
  hub.position.z = 0.4 * MM;
  g.add(hub);

  const arms = doubleArm ? [-1, 1] : [1];
  for (const dir of arms) {
    const armGeo = new THREE.BoxGeometry(armLenMM * MM, 4.6 * MM, 1.8 * MM);
    const arm = new THREE.Mesh(armGeo, hornMat);
    arm.position.set(dir * (armLenMM * 0.5) * MM, 0, 1.1 * MM);
    g.add(arm);

    const tip = new THREE.Mesh(new THREE.CylinderGeometry(2.1 * MM, 2.1 * MM, 1.8 * MM, 14), hornMat);
    tip.rotation.x = Math.PI / 2;
    tip.position.set(dir * armLenMM * MM, 0, 1.1 * MM);
    g.add(tip);

    // Radial horn locking screw
    const rScrew = createM2PanScrew(mats, 3.6);
    rScrew.position.set(dir * (armLenMM - 2.2) * MM, 0, 2.0 * MM);
    g.add(rScrew);
  }

  // Central M2x5 spline retaining machine screw
  const centerScrew = createM2PanScrew(mats, 4.8);
  centerScrew.position.z = 2.1 * MM;
  g.add(centerScrew);

  return { group: g, centerScrew };
}

/**
 * Ultra-Detailed MG90S (Metal-Gear) / SG90 (Translucent Nylon-Gear) Micro Servo.
 * CRITICAL KINEMATIC CONVENTION:
 *   - Local Origin (0, 0, 0) = EXACT Output Spline Joint Axis at the top bearing face!
 *   - Output 21T Brass/Nylon Spline points along +Z (0 .. +3.2 mm).
 *   - Housing body center is offset along -X by 5.5 mm (22.8/2 - 5.9 = 5.5 mm),
 *     guaranteeing 0.00 mm coaxial alignment on all 12 robot joints!
 *   - Includes full INTERNAL 4-stage reduction gear train + coreless DC motor + potentiometer + driver PCB!
 */
export function createPrecisionServo(mats, isSG90 = false) {
  const root = new THREE.Group();
  const caseMat = isSG90 ? mats.servoSG90 : mats.servoMG90S;
  const gearMat = isSG90 ? mats.nylonWhite : mats.brass;
  const bodyL = (isSG90 ? 22.5 : 22.8) * MM;
  const bodyW = (isSG90 ? 12.0 : 12.2) * MM;
  const bodyH = (isSG90 ? 19.5 : 21.5) * MM;
  const xOff = -5.5 * MM; // Exact shaft-to-body-center offset

  // 1. Outer Housing (clipped cleanly in Section Cutaway mode!)
  const housingGroup = new THREE.Group();
  root.add(housingGroup);

  const mainCase = new THREE.Mesh(new THREE.BoxGeometry(bodyL, bodyW, bodyH), caseMat);
  mainCase.position.set(xOff, 0, -3.5 * MM - bodyH * 0.5);
  mainCase.castShadow = true;
  housingGroup.add(mainCase);

  // Upper gearbox step + output bearing boss
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

  const auxDome = new THREE.Mesh(new THREE.CylinderGeometry(2.9 * MM, 2.9 * MM, 2.4 * MM, 16), caseMat);
  auxDome.rotation.x = Math.PI / 2;
  auxDome.position.set(-5.8 * MM, 0, -1.0 * MM);
  housingGroup.add(auxDome);

  // Dual Mounting Ears (32.2 mm span, 27.6 mm M2 hole centers)
  const flange = new THREE.Mesh(new THREE.BoxGeometry(32.2 * MM, bodyW, 2.4 * MM), caseMat);
  flange.position.set(xOff, 0, -8.2 * MM);
  housingGroup.add(flange);

  for (const dx of [-13.8, 13.8]) {
    const eyelet = new THREE.Mesh(new THREE.CylinderGeometry(1.7 * MM, 1.7 * MM, 2.6 * MM, 14), mats.brass);
    eyelet.rotation.x = Math.PI / 2;
    eyelet.position.set(xOff + dx * MM, 0, -8.2 * MM);
    root.add(eyelet);

    const mScrew = createM2PanScrew(mats, 5.5);
    mScrew.position.set(xOff + dx * MM, 0, -6.9 * MM);
    root.add(mScrew);
  }

  // 2. Output 21-Tooth Spline (Uncut!)
  const spline = new THREE.Mesh(
    new THREE.CylinderGeometry(2.45 * MM, 2.45 * MM, 3.4 * MM, 20),
    isSG90 ? mats.nylonWhite : mats.brass
  );
  spline.rotation.x = Math.PI / 2;
  spline.position.set(0, 0, 1.6 * MM);
  root.add(spline);

  // 3. Internal Anatomy: Coreless DC Motor + 4-Stage Spur Gear Train + Potentiometer + Driver PCB!
  const internals = new THREE.Group();
  root.add(internals);

  // Coreless Brushed DC Motor (Ø9.8 x 13.5 mm steel can)
  const motorCan = new THREE.Mesh(new THREE.CylinderGeometry(4.8 * MM, 4.8 * MM, 13.2 * MM, 20), mats.stainless);
  motorCan.rotation.x = Math.PI / 2;
  motorCan.position.set(-10.6 * MM, 0, -15.5 * MM);
  internals.add(motorCan);

  const motorEndbell = new THREE.Mesh(new THREE.CylinderGeometry(4.6 * MM, 4.6 * MM, 1.6 * MM, 16), mats.copper);
  motorEndbell.rotation.x = Math.PI / 2;
  motorEndbell.position.set(-10.6 * MM, 0, -22.2 * MM);
  internals.add(motorEndbell);

  // Stage 0: Motor Brass Pinion
  const pinion = new THREE.Mesh(new THREE.CylinderGeometry(1.3 * MM, 1.3 * MM, 3.2 * MM, 12), mats.brass);
  pinion.rotation.x = Math.PI / 2;
  pinion.position.set(-10.6 * MM, 0, -7.2 * MM);
  internals.add(pinion);

  // Stage 1..3: Stepped Spur Reduction Gears + Stainless Axles
  const gearSpecs = [
    { x: -8.2, y:  1.6, z: -6.8, rBig: 3.6, rPin: 1.3 },
    { x: -5.2, y: -1.5, z: -5.2, rBig: 4.1, rPin: 1.5 },
    { x:  0.0, y:  0.0, z: -3.4, rBig: 5.1, rPin: 2.2 }, // Main output spur gear concentric with shaft!
  ];
  for (const gs of gearSpecs) {
    const gBig = new THREE.Mesh(
      new THREE.CylinderGeometry(gs.rBig * MM, gs.rBig * MM, 1.5 * MM, 20),
      gearMat
    );
    gBig.rotation.x = Math.PI / 2;
    gBig.position.set(gs.x * MM, gs.y * MM, gs.z * MM);
    internals.add(gBig);

    const gSub = new THREE.Mesh(
      new THREE.CylinderGeometry(gs.rPin * MM, gs.rPin * MM, 2.8 * MM, 14),
      mats.darkSteel
    );
    gSub.rotation.x = Math.PI / 2;
    gSub.position.set(gs.x * MM, gs.y * MM, (gs.z - 0.8) * MM);
    internals.add(gSub);
  }

  // Feedback Rotary Potentiometer beneath output gear
  const potBody = new THREE.Mesh(new THREE.CylinderGeometry(3.8 * MM, 3.8 * MM, 4.6 * MM, 16), mats.anodizedBlack);
  potBody.rotation.x = Math.PI / 2;
  potBody.position.set(0, 0, -10.2 * MM);
  internals.add(potBody);

  // Internal H-Bridge Servo Control PCB
  const ctrlPcb = new THREE.Mesh(new THREE.BoxGeometry(9.5 * MM, 9.8 * MM, 1.2 * MM), mats.pcbGreen);
  ctrlPcb.position.set(-1.2 * MM, 0, -19.5 * MM);
  internals.add(ctrlPcb);
  const driverIc = new THREE.Mesh(new THREE.BoxGeometry(3.6 * MM, 3.0 * MM, 1.0 * MM), mats.icEpoxy);
  driverIc.position.set(-1.2 * MM, 0, -18.5 * MM);
  internals.add(driverIc);

  // Strain-relief 3-wire JR Silicone Ribbon Boot (Brown GND / Red 5.6V / Amber PWM)
  const boot = new THREE.Mesh(new THREE.BoxGeometry(3.2 * MM, 4.4 * MM, 2.8 * MM), mats.soleSilicone);
  boot.position.set(xOff - bodyL * 0.5 - 1.2 * MM, 0, -21.0 * MM);
  internals.add(boot);

  return root;
}

/**
 * Seeed Studio XIAO ESP32S3 Sense (21.0 x 17.8 x 15.0 mm, 6.5 g)
 * Includes: Main PCB, 14 gold castellated pads, USB-C port, ESP32-S3R8 RF shield,
 * 25-pin CNC Black Anodized Heatsink, B2B Sense Expansion Board, MSM261 PDM Digital Mic,
 * MicroSD cage, U.FL + FPC 2.4GHz Antenna, Golden 24-pin DVP Flex, and Multi-Element OV5640 Lens!
 */
export function createXiaoESP32S3SenseAssembly(mats) {
  const g = new THREE.Group();

  // 1. Main XIAO ESP32S3 4-Layer Matte-Black PCB (21.0 mm Z x 17.8 mm Y x 1.6 mm X)
  const mainPcb = new THREE.Mesh(new THREE.BoxGeometry(1.6 * MM, 17.8 * MM, 21.0 * MM), mats.pcbBlack);
  g.add(mainPcb);

  // 14x Gold Castellated Half-Holes (7 per side, 2.54 mm pitch: D0..D6 & 5V,GND,3V3,D10..D7)
  for (const side of [-1, 1]) {
    for (let i = -3; i <= 3; ++i) {
      const pad = new THREE.Mesh(new THREE.CylinderGeometry(0.85 * MM, 0.85 * MM, 1.75 * MM, 12), mats.gold);
      pad.rotation.z = Math.PI / 2;
      pad.position.set(0, side * 8.6 * MM, i * 2.54 * MM);
      g.add(pad);
    }
  }

  // Top Rolled Stainless USB Type-C Receptacle (facing +Z inside upper cranium)
  const usbShell = new THREE.Mesh(new THREE.BoxGeometry(3.2 * MM, 8.94 * MM, 7.3 * MM), mats.stainless);
  usbShell.position.set(2.2 * MM, 0, 8.2 * MM);
  g.add(usbShell);
  const usbTongue = new THREE.Mesh(new THREE.BoxGeometry(0.7 * MM, 6.6 * MM, 4.0 * MM), mats.icEpoxy);
  usbTongue.position.set(2.2 * MM, 0, 9.8 * MM);
  g.add(usbTongue);

  // ESP32-S3R8 Nickel RF Shield Can on rear (-X) + CNC 5x5 Pin-Fin Anodized Aluminum Heatsink
  const rfShield = new THREE.Mesh(new THREE.BoxGeometry(2.2 * MM, 12.2 * MM, 12.2 * MM), mats.stainless);
  rfShield.position.set(-1.9 * MM, 0, -1.5 * MM);
  g.add(rfShield);

  const hsBase = new THREE.Mesh(new THREE.BoxGeometry(1.6 * MM, 13.6 * MM, 13.6 * MM), mats.anodizedBlack);
  hsBase.position.set(-3.8 * MM, 0, -1.5 * MM);
  g.add(hsBase);

  // 25 individual CNC cooling pins aligned with head_shell.scad rear chimney exhaust vents!
  for (let iy = -2; iy <= 2; ++iy) {
    for (let iz = -2; iz <= 2; ++iz) {
      const fin = new THREE.Mesh(GEO_HEATSINK_PIN, mats.anodizedBlack);
      fin.position.set(-6.2 * MM, iy * 2.6 * MM, (-1.5 + iz * 2.6) * MM);
      g.add(fin);
    }
  }

  // 2. High-Density B2B Connector + Sense Expansion Daughterboard (+X side)
  const b2b = new THREE.Mesh(new THREE.BoxGeometry(2.8 * MM, 10.0 * MM, 4.2 * MM), mats.icEpoxy);
  b2b.position.set(2.2 * MM, 0, -4.5 * MM);
  g.add(b2b);

  const sensePcb = new THREE.Mesh(new THREE.BoxGeometry(1.6 * MM, 17.8 * MM, 21.0 * MM), mats.pcbBlack);
  sensePcb.position.set(4.4 * MM, 0, 0);
  g.add(sensePcb);

  // Onboard MSM261D3526H1CPM PDM Digital Microphone (aligned with head_shell acoustic ports!)
  const pdmMic = new THREE.Mesh(new THREE.BoxGeometry(1.1 * MM, 3.5 * MM, 2.65 * MM), mats.stainless);
  pdmMic.position.set(5.6 * MM, 0, 7.6 * MM);
  g.add(pdmMic);
  const micPort = new THREE.Mesh(new THREE.CylinderGeometry(0.55 * MM, 0.55 * MM, 1.2 * MM, 12), mats.gold);
  micPort.rotation.z = Math.PI / 2;
  micPort.position.set(5.7 * MM, 0, 7.6 * MM);
  g.add(micPort);

  // MicroSD Card Stainless Slot on Sense board
  const sdSlot = new THREE.Mesh(new THREE.BoxGeometry(1.8 * MM, 11.5 * MM, 11.0 * MM), mats.stainless);
  sdSlot.position.set(2.8 * MM, 0, 2.0 * MM);
  g.add(sdSlot);

  // 3. OV5640 / OV3660 Camera Sensor + Golden 24-pin DVP FPC Ribbon + Multi-Coated Objective Lens
  const fpcRibbon = new THREE.Mesh(new THREE.BoxGeometry(0.35 * MM, 8.0 * MM, 11.5 * MM), mats.fpcGold);
  fpcRibbon.position.set(5.5 * MM, 0, -2.5 * MM);
  g.add(fpcRibbon);

  const camBase = new THREE.Mesh(new THREE.BoxGeometry(3.2 * MM, 8.5 * MM, 8.5 * MM), mats.anodizedBlack);
  camBase.position.set(6.8 * MM, 0, 0);
  g.add(camBase);

  const lensBarrel = new THREE.Mesh(new THREE.CylinderGeometry(4.0 * MM, 4.2 * MM, 4.6 * MM, 28), mats.anodizedBlack);
  lensBarrel.rotation.z = Math.PI / 2;
  lensBarrel.position.set(9.4 * MM, 0, 0);
  g.add(lensBarrel);

  const lensBezel = new THREE.Mesh(new THREE.TorusGeometry(3.3 * MM, 0.45 * MM, 12, 24), mats.aluminum);
  lensBezel.rotation.y = Math.PI / 2;
  lensBezel.position.set(11.6 * MM, 0, 0);
  g.add(lensBezel);

  const lensGlass = new THREE.Mesh(new THREE.SphereGeometry(2.85 * MM, 24, 16, 0, Math.PI * 2, 0, Math.PI * 0.42), mats.lensGlass);
  lensGlass.rotation.z = -Math.PI / 2;
  lensGlass.position.set(10.8 * MM, 0, 0);
  g.add(lensGlass);

  // 4. U.FL Coax + 2.4 GHz Wi-Fi/BLE FPC Dipole Antenna along cranium inner wall (+Y)
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
 * VL53L0X (GY-530) 940 nm Laser Time-of-Flight Sensor (13.3 x 10.5 x 3.0 mm, I2C 0x29).
 * Shows distinct dual optical cavities: VCSEL 940nm emitter + SPAD photon detector array!
 */
export function createVL53L0XAssembly(mats) {
  const g = new THREE.Group();
  const pcb = new THREE.Mesh(new THREE.BoxGeometry(1.6 * MM, 13.3 * MM, 10.5 * MM), mats.pcbPurple);
  g.add(pcb);

  // 2x M2 Mounting Holes with Gold Annular Rings + M2 Screws
  for (const sy of [-4.8, 4.8]) {
    const pad = new THREE.Mesh(new THREE.CylinderGeometry(1.6 * MM, 1.6 * MM, 1.7 * MM, 14), mats.gold);
    pad.rotation.z = Math.PI / 2;
    pad.position.set(0, sy * MM, -3.2 * MM);
    g.add(pad);
  }

  // Optical LGA-12 VL53L0X Package (4.4 x 2.4 x 1.1 mm) aligned at Z=0
  const lgaCap = new THREE.Mesh(new THREE.BoxGeometry(1.2 * MM, 4.4 * MM, 2.4 * MM), mats.icEpoxy);
  lgaCap.position.set(1.3 * MM, 0, 0);
  g.add(lgaCap);

  // Dual Optical Apertures (VCSEL Emitter + SPAD Receiver)
  for (const dy of [-1.05, 1.05]) {
    const optic = new THREE.Mesh(new THREE.CylinderGeometry(0.65 * MM, 0.65 * MM, 0.4 * MM, 14), mats.tofWindow);
    optic.rotation.z = Math.PI / 2;
    optic.position.set(1.85 * MM, dy * MM, 0);
    g.add(optic);
  }

  // SOT-23-5 2.8V LDO + I2C Level-Shifter MOSFETs
  const ldo = new THREE.Mesh(new THREE.BoxGeometry(1.0 * MM, 2.9 * MM, 1.6 * MM), mats.icEpoxy);
  ldo.position.set(-1.1 * MM, -2.5 * MM, 2.2 * MM);
  g.add(ldo);
  for (const dy of [-2.2, 2.2]) {
    const cap = new THREE.Mesh(GEO_SMD_0603, mats.ceramicCap);
    cap.position.set(-1.0 * MM, dy * MM, -0.5 * MM);
    g.add(cap);
  }

  // 6-Pin Right-Angle Header on rear (-X)
  for (let i = 0; i < 6; ++i) {
    const pin = new THREE.Mesh(GEO_PIN_GOLD, mats.gold);
    pin.rotation.y = Math.PI / 2;
    pin.position.set(-3.2 * MM, (-6.35 + i * 2.54) * MM, 3.8 * MM);
    g.add(pin);
  }

  return g;
}

/**
 * PCA9685 16-Channel 12-Bit I2C PWM Servo Driver (62.0 x 25.0 x 12.0 mm, 8.5 g, I2C 0x40).
 * Oriented in Y-Z plane inside Torso: Y = -31..+31 mm, Z = -12.5..+12.5 mm, headers facing +X!
 */
export function createPCA9685Assembly(mats) {
  const g = new THREE.Group();

  const pcb = new THREE.Mesh(new THREE.BoxGeometry(1.6 * MM, 62.0 * MM, 25.0 * MM), mats.pcbBlue);
  pcb.castShadow = true;
  g.add(pcb);

  // 4x Corner M2 Mounting Standoffs + Stainless Bolts
  for (const sy of [-28.5, 28.5]) {
    for (const sz of [-10.0, 10.0]) {
      const pad = new THREE.Mesh(new THREE.CylinderGeometry(2.1 * MM, 2.1 * MM, 1.7 * MM, 14), mats.gold);
      pad.rotation.z = Math.PI / 2;
      pad.position.set(0, sy * MM, sz * MM);
      g.add(pad);

      const bolt = createM2PanScrew(mats, 4.5);
      bolt.rotation.y = Math.PI / 2;
      bolt.position.set(1.0 * MM, sy * MM, sz * MM);
      g.add(bolt);
    }
  }

  // Central NXP PCA9685PW TSSOP-28 IC + Metallic Gull-Wing Leadframe
  const icBody = new THREE.Mesh(new THREE.BoxGeometry(1.2 * MM, 9.7 * MM, 4.4 * MM), mats.icEpoxy);
  icBody.position.set(1.3 * MM, 0, 2.2 * MM);
  g.add(icBody);
  const icLeads = new THREE.Mesh(new THREE.BoxGeometry(0.4 * MM, 9.4 * MM, 6.4 * MM), mats.stainless);
  icLeads.position.set(1.0 * MM, 0, 2.2 * MM);
  g.add(icLeads);

  // 4x Banks of 3x4 Color-Coded 2.54mm Servo Header Strips (PWM=Yellow, V+=Red, GND=Black) + 48 Gold Pins!
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
      // Channels CH0..CH11 have mated 3-pin servo DuPont housings plugged in!
      if (chIndex < 12) {
        const plug = new THREE.Mesh(new THREE.BoxGeometry(5.6 * MM, 2.35 * MM, 7.5 * MM), mats.hdrBlack);
        plug.position.set(6.2 * MM, py, -6.14 * MM);
        g.add(plug);
      }
      chIndex++;
    }
  }

  // Top-Center 2-Pin Green Pluggable Screw Terminal Block (V+ 5.6V & GND)
  const termBlock = new THREE.Mesh(new THREE.BoxGeometry(7.5 * MM, 10.2 * MM, 8.0 * MM), mats.termGreen);
  termBlock.position.set(4.5 * MM, 0, 8.2 * MM);
  g.add(termBlock);
  for (const sy of [-2.54, 2.54]) {
    const screw = new THREE.Mesh(new THREE.CylinderGeometry(1.7 * MM, 1.7 * MM, 1.2 * MM, 14), mats.stainless);
    screw.position.set(5.2 * MM, sy * MM, 12.1 * MM);
    g.add(screw);
  }

  // Onboard Radial Bulk Decoupling Cap + 10k OE# Pull-Up SMD Array + Side I2C Header
  const cap = new THREE.Mesh(new THREE.CylinderGeometry(3.2 * MM, 3.2 * MM, 7.0 * MM, 18), mats.capSleeve);
  cap.rotation.z = Math.PI / 2;
  cap.position.set(4.3 * MM, -14.5 * MM, 7.8 * MM);
  g.add(cap);
  const capTop = new THREE.Mesh(new THREE.CylinderGeometry(3.0 * MM, 3.0 * MM, 0.3 * MM, 18), mats.aluminum);
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
 * MPU-6050 (GY-521) 6-Axis MEMS IMU (15.8 x 20.5 x 3.2 mm, I2C 0x68) mounted on Pelvis CoM Cross-Tie Beam!
 * Includes QFN-24 MEMS die, SOT-23-5 LDO, 2x M2x5 mounting screws at Y=+-8.0mm, and 3D IMU/CoM Triad!
 */
export function createMPU6050CoMAssembly(mats) {
  const g = new THREE.Group();

  const pcb = new THREE.Mesh(new THREE.BoxGeometry(15.8 * MM, 20.5 * MM, 1.6 * MM), mats.pcbBlue);
  g.add(pcb);

  // QFN-24 InvenSense MPU-6050 MEMS Die at exact center
  const qfn = new THREE.Mesh(new THREE.BoxGeometry(4.0 * MM, 4.0 * MM, 0.95 * MM), mats.icEpoxy);
  qfn.position.set(0, 0, 1.25 * MM);
  g.add(qfn);
  const qfnPads = new THREE.Mesh(new THREE.BoxGeometry(4.5 * MM, 4.5 * MM, 0.3 * MM), mats.stainless);
  qfnPads.position.set(0, 0, 0.92 * MM);
  g.add(qfnPads);

  // 3.3V SOT-23-5 LDO + SMD Capacitors
  const ldo = new THREE.Mesh(new THREE.BoxGeometry(2.9 * MM, 1.6 * MM, 1.0 * MM), mats.icEpoxy);
  ldo.position.set(-4.2 * MM, 3.8 * MM, 1.2 * MM);
  g.add(ldo);
  for (const sy of [-4.5, -2.0, 5.5]) {
    const smd = new THREE.Mesh(GEO_SMD_0603, mats.ceramicCap);
    smd.position.set(3.8 * MM, sy * MM, 1.1 * MM);
    g.add(smd);
  }

  // 2x M2x5 Stainless Mounting Screws into torso_chassis.scad pilot holes (Y = +-8.0 mm)!
  for (const sy of [-8.0, 8.0]) {
    const screw = createM2PanScrew(mats, 5.2);
    screw.position.set(0, sy * MM, 0.85 * MM);
    g.add(screw);
  }

  // 8-Pin Header Row along -X edge
  for (let i = 0; i < 4; ++i) {
    const pin = new THREE.Mesh(GEO_PIN_GOLD, mats.gold);
    pin.position.set(-6.4 * MM, (-3.81 + i * 2.54) * MM, 2.4 * MM);
    g.add(pin);
  }

  return g;
}

/**
 * Dual-Rail Isolated Power System (Mini560 5.6V 5A Servo Buck + Mini-360 5.0V 2A Logic Buck
 * + 1000uF 10V Low-ESR Electrolytic Bulk Cap + 1N5819 Reverse-Blocking Schottky + VBAT Divider).
 */
export function createDualBuckPowerNodes(mats) {
  // 1. Mini560 5.6V 5A Synchronous Buck Converter (30.0 x 18.0 mm, vertical along -Y side louvers)
  const buck5A = new THREE.Group();
  const pcb5A = new THREE.Mesh(new THREE.BoxGeometry(18.0 * MM, 2.0 * MM, 30.0 * MM), mats.pcbGreen);
  buck5A.add(pcb5A);

  // Shielded 6.8x6.8x4.0mm Molded Alloy Power Inductor ("4R7")
  const ind5A = new THREE.Mesh(new THREE.BoxGeometry(7.0 * MM, 4.2 * MM, 7.0 * MM), mats.ferrite);
  ind5A.position.set(0.5 * MM, 2.8 * MM, -2.0 * MM);
  buck5A.add(ind5A);

  // Synchronous Buck QFN IC + 6x 1206 MLCC Input/Output Ceramic Caps + Trim Pot
  const ic5A = new THREE.Mesh(new THREE.BoxGeometry(4.0 * MM, 1.1 * MM, 4.0 * MM), mats.icEpoxy);
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

  // 2. 1000 uF 10V Low-ESR Radial Electrolytic Bulk Capacitor (Ø8.0 x 12.0 mm)
  const cap1000 = new THREE.Group();
  const capCan = new THREE.Mesh(new THREE.CylinderGeometry(4.2 * MM, 4.2 * MM, 12.0 * MM, 24), mats.capSleeve);
  capCan.rotation.x = Math.PI / 2;
  cap1000.add(capCan);
  const capVent = new THREE.Mesh(new THREE.CylinderGeometry(3.9 * MM, 3.9 * MM, 0.35 * MM, 24), mats.aluminum);
  capVent.rotation.x = Math.PI / 2;
  capVent.position.z = 6.05 * MM;
  cap1000.add(capVent);
  const capStripe = new THREE.Mesh(new THREE.BoxGeometry(1.4 * MM, 8.5 * MM, 11.8 * MM), mats.aluminum);
  capStripe.position.x = -0.1 * MM;
  capStripe.scale.set(0.18, 0.99, 0.98);
  cap1000.add(capStripe);
  for (const dx of [-1.75, 1.75]) {
    const lead = new THREE.Mesh(new THREE.CylinderGeometry(0.35 * MM, 0.35 * MM, 4.5 * MM, 8), mats.stainless);
    lead.rotation.x = Math.PI / 2;
    lead.position.set(dx * MM, 0, -7.8 * MM);
    cap1000.add(lead);
  }

  // 3. Mini-360 5.0V 2A Isolated Logic Buck + 1N5819 Schottky Diode + 200k/100k VBAT Divider (+Y side)
  const buck2A = new THREE.Group();
  const pcb2A = new THREE.Mesh(new THREE.BoxGeometry(11.2 * MM, 1.8 * MM, 17.5 * MM), mats.pcbGreen);
  buck2A.add(pcb2A);

  const ind2A = new THREE.Mesh(new THREE.BoxGeometry(4.4 * MM, 3.0 * MM, 4.4 * MM), mats.ferrite);
  ind2A.position.set(0, -2.2 * MM, -2.5 * MM);
  buck2A.add(ind2A);

  const ic2A = new THREE.Mesh(new THREE.BoxGeometry(4.9 * MM, 1.4 * MM, 3.9 * MM), mats.icEpoxy);
  ic2A.position.set(0.5 * MM, -1.5 * MM, 4.0 * MM);
  buck2A.add(ic2A);

  const pot2A = new THREE.Mesh(new THREE.CylinderGeometry(1.8 * MM, 1.8 * MM, 1.4 * MM, 14), mats.brass);
  pot2A.position.set(-2.5 * MM, -1.5 * MM, -5.8 * MM);
  buck2A.add(pot2A);

  // 1N5819 Schottky Barrier Diode (DO-41 axial package w/ silver cathode band -> prevents USB backfeed!)
  const schottkyBody = new THREE.Mesh(new THREE.CylinderGeometry(1.35 * MM, 1.35 * MM, 4.8 * MM, 16), mats.icEpoxy);
  schottkyBody.position.set(0, -2.6 * MM, 11.8 * MM);
  schottkyBody.rotation.x = Math.PI / 2;
  buck2A.add(schottkyBody);
  const cathodeBand = new THREE.Mesh(new THREE.CylinderGeometry(1.40 * MM, 1.40 * MM, 0.9 * MM, 16), mats.aluminum);
  cathodeBand.rotation.x = Math.PI / 2;
  cathodeBand.position.set(0, -2.6 * MM, 13.3 * MM);
  buck2A.add(cathodeBand);

  // 200kOhm / 100kOhm Precision VBAT_ADC Divider Resistors
  for (const dz of [9.5, 11.2]) {
    const rDiv = new THREE.Mesh(GEO_SMD_0603, mats.icEpoxy);
    rDiv.position.set(3.6 * MM, -1.4 * MM, dz * MM);
    buck2A.add(rDiv);
  }

  return { buck5A, cap1000, buck2A };
}

/**
 * MAX98357A I2S 3W Class-D Amplifier + 28 mm 4ohm 3W Cavity Neodymium Speaker (11.5 g).
 */
export function createAudioSystemNodes(mats) {
  // 1. MAX98357A I2S Breakout PCB (19.0 x 17.8 x 1.6 mm)
  const ampGroup = new THREE.Group();
  const pcb = new THREE.Mesh(new THREE.BoxGeometry(17.8 * MM, 1.6 * MM, 19.0 * MM), mats.pcbPurple);
  ampGroup.add(pcb);

  const qfn = new THREE.Mesh(new THREE.BoxGeometry(3.2 * MM, 1.0 * MM, 3.2 * MM), mats.icEpoxy);
  qfn.position.set(0, -1.2 * MM, 0);
  ampGroup.add(qfn);

  const spkTerm = new THREE.Mesh(new THREE.BoxGeometry(6.5 * MM, 6.2 * MM, 5.2 * MM), mats.termGreen);
  spkTerm.position.set(3.5 * MM, -3.6 * MM, -6.2 * MM);
  ampGroup.add(spkTerm);

  for (let i = 0; i < 7; ++i) {
    const pin = new THREE.Mesh(GEO_PIN_GOLD, mats.gold);
    pin.position.set((-6.5 + i * 2.1) * MM, -2.2 * MM, 7.5 * MM);
    ampGroup.add(pin);
  }

  // 2. 28 mm 4ohm 3W Neodymium Speaker seated inside torso_chassis front acoustic ring!
  const spkGroup = new THREE.Group();
  // Outer stamped basket rim (Ø28.0 mm)
  const rim = new THREE.Mesh(new THREE.CylinderGeometry(14.0 * MM, 12.8 * MM, 2.2 * MM, 32), mats.anodizedBlack);
  rim.rotation.z = Math.PI / 2;
  rim.position.x = 1.2 * MM;
  spkGroup.add(rim);

  // Corrugated Half-Roll Rubber Surround (Torus at +X front face)
  const surround = new THREE.Mesh(new THREE.TorusGeometry(11.4 * MM, 1.35 * MM, 12, 32), mats.soleSilicone);
  surround.rotation.y = Math.PI / 2;
  surround.position.x = 2.0 * MM;
  spkGroup.add(surround);

  // Concave Acoustic Diaphragm Cone + Central Voice-Coil Dust Cap
  const cone = new THREE.Mesh(new THREE.ConeGeometry(10.6 * MM, 3.4 * MM, 32, 1, true), mats.speakerCone);
  cone.rotation.z = Math.PI / 2; // Apex at -X, open mouth at +X
  cone.position.x = 0.5 * MM;
  spkGroup.add(cone);

  const dustCap = new THREE.Mesh(
    new THREE.SphereGeometry(4.2 * MM, 20, 12, 0, Math.PI * 2, 0, Math.PI * 0.45),
    mats.anodizedBlack
  );
  dustCap.rotation.z = -Math.PI / 2;
  dustCap.position.x = -0.2 * MM;
  spkGroup.add(dustCap);

  // Rear Basket Spider Struts + Shielded Neodymium Pot Magnet + Gold Solder Lugs
  const basket = new THREE.Mesh(new THREE.CylinderGeometry(11.5 * MM, 8.2 * MM, 2.8 * MM, 24), mats.darkSteel);
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
 * 2S 14500 800mAh 10C Li-ion Pack + 2S 5A BMS + USB-C Boost Charger + 6A Main Slide Switch (51.0 g).
 * Exact alignment with torso_back_cover.scad cutouts:
 *   - USB-C 2S Charging Port at (X = -24.5 mm, Y = -14.0 mm, Z = +14.5 mm)
 *   - 6A Main Slide Switch at (X = -24.5 mm, Y = +14.0 mm, Z = +14.5 mm)
 */
export function createBattery2SAndRearIO(mats) {
  const packGroup = new THREE.Group();

  // Contoured Dual-14500 ABS Holder Cradle
  const cradle = new THREE.Mesh(new THREE.BoxGeometry(8.0 * MM, 31.5 * MM, 52.4 * MM), mats.anodizedBlack);
  cradle.position.set(3.2 * MM, 0, 0);
  packGroup.add(cradle);

  const cells = [];
  for (const sy of [-8.0, 8.0]) {
    const cellGrp = new THREE.Group();
    cellGrp.position.set(0, sy * MM, 0);

    const wrap = new THREE.Mesh(new THREE.CylinderGeometry(7.0 * MM, 7.0 * MM, 49.2 * MM, 28), mats.liIonBlue);
    wrap.rotation.x = Math.PI / 2;
    wrap.castShadow = true;
    cellGrp.add(wrap);

    // White insulating gasket + copper/nickel button-top positive terminal (alternating series polarity!)
    const polDir = sy < 0 ? 1 : -1;
    const gasket = new THREE.Mesh(new THREE.CylinderGeometry(6.6 * MM, 6.6 * MM, 0.6 * MM, 20), mats.nylonWhite);
    gasket.rotation.x = Math.PI / 2;
    gasket.position.z = polDir * 24.8 * MM;
    cellGrp.add(gasket);

    const posCap = new THREE.Mesh(new THREE.CylinderGeometry(3.8 * MM, 4.2 * MM, 1.4 * MM, 18), mats.stainless);
    posCap.rotation.x = Math.PI / 2;
    posCap.position.z = polDir * 25.4 * MM;
    cellGrp.add(posCap);

    const negBase = new THREE.Mesh(new THREE.CylinderGeometry(6.4 * MM, 6.4 * MM, 0.5 * MM, 20), mats.stainless);
    negBase.rotation.x = Math.PI / 2;
    negBase.position.z = -polDir * 24.8 * MM;
    cellGrp.add(negBase);

    packGroup.add(cellGrp);
    cells.push(cellGrp);
  }

  // Spot-Welded 0.15mm Pure Nickel Series Busbar Strip (BM link at top Z = +25.9 mm)
  const nickelSeries = new THREE.Mesh(new THREE.BoxGeometry(4.5 * MM, 21.0 * MM, 0.4 * MM), mats.stainless);
  nickelSeries.position.set(0, 0, 25.9 * MM);
  packGroup.add(nickelSeries);

  // 2S 5A BMS Protection PCB (28.0 x 10.0 x 2.0 mm) with 4x 8205A Dual MOSFETs + R010 Shunt
  const bmsGroup = new THREE.Group();
  bmsGroup.position.set(-8.2 * MM, 0, 2.0 * MM);
  const bmsPcb = new THREE.Mesh(new THREE.BoxGeometry(1.8 * MM, 28.0 * MM, 10.5 * MM), mats.pcbGreen);
  bmsGroup.add(bmsPcb);
  for (const dy of [-9.0, -3.5, 2.0, 7.5]) {
    const mosfet = new THREE.Mesh(new THREE.BoxGeometry(1.1 * MM, 3.2 * MM, 4.4 * MM), mats.icEpoxy);
    mosfet.position.set(-1.3 * MM, dy * MM, 0);
    bmsGroup.add(mosfet);
  }
  packGroup.add(bmsGroup);

  // Rear Service Panel IO Group (USB-C 2S Boost Charger & 6A Main Slide Switch aligned with cover cutouts!)
  const rearIOGroup = new THREE.Group();

  // USB-C 2S Boost Charger at (X = -22.0, Y = -14.0, Z = +14.5 mm in torso frame)
  const usbcPcb = new THREE.Mesh(new THREE.BoxGeometry(6.5 * MM, 11.0 * MM, 2.0 * MM), mats.pcbBlack);
  usbcPcb.position.set(-19.8 * MM, -14.0 * MM, 12.6 * MM);
  rearIOGroup.add(usbcPcb);
  const usbcPort = new THREE.Mesh(new THREE.BoxGeometry(7.2 * MM, 8.94 * MM, 3.2 * MM), mats.stainless);
  usbcPort.position.set(-22.2 * MM, -14.0 * MM, 14.5 * MM);
  rearIOGroup.add(usbcPort);
  const usbcCav = new THREE.Mesh(new THREE.BoxGeometry(1.2 * MM, 7.2 * MM, 1.8 * MM), mats.icEpoxy);
  usbcCav.position.set(-25.4 * MM, -14.0 * MM, 14.5 * MM);
  rearIOGroup.add(usbcCav);

  // 6A Panel-Mount Main Slide Switch at (X = -22.0, Y = +14.0, Z = +14.5 mm in torso frame)
  const swBody = new THREE.Mesh(new THREE.BoxGeometry(5.5 * MM, 11.2 * MM, 5.0 * MM), mats.stainless);
  swBody.position.set(-21.2 * MM, 14.0 * MM, 14.5 * MM);
  rearIOGroup.add(swBody);
  const swKnob = new THREE.Mesh(new THREE.BoxGeometry(3.4 * MM, 4.2 * MM, 3.4 * MM), mats.anodizedBlack);
  swKnob.position.set(-25.0 * MM, 15.6 * MM, 14.5 * MM);
  rearIOGroup.add(swKnob);
  for (const dy of [-3.5, 0, 3.5]) {
    const lug = new THREE.Mesh(new THREE.BoxGeometry(3.2 * MM, 0.8 * MM, 1.4 * MM), mats.silver || mats.stainless);
    lug.position.set(-17.2 * MM, (14.0 + dy) * MM, 14.5 * MM);
    rearIOGroup.add(lug);
  }

  return { packGroup, cells, bmsGroup, rearIOGroup };
}

/**
 * Helper to create a smooth 3D Catmull-Rom silicone wire tube from mm waypoints.
 */
export function createWireTube(ptsMM, radiusMM, mat, tubularSegments = 28) {
  const vecs = ptsMM.map((p) => new THREE.Vector3(p[0] * MM, p[1] * MM, p[2] * MM));
  const curve = new THREE.CatmullRomCurve3(vecs);
  const geo = new THREE.TubeGeometry(curve, tubularSegments, radiusMM * MM, 8, false);
  return new THREE.Mesh(geo, mat);
}

/**
 * Build Complete Real-Hardware 3D Silicone Wiring Harness:
 *   - Torso Power & Data Buses (VBAT 7.4V, 5.6V Servo, 5.0V Logic via 1N5819, 400kHz I2C, I2S + Speaker)
 *   - Head Coaxial I2C + Power Flex Harness (XIAO <-> VL53L0X & Neck Umbilical)
 *   - Left & Right Leg Servo Harnesses routed directly through leg_link.scad's X = -6.8mm rear conduit!
 */
export function createCompleteWiringHarnesses(mats) {
  const torsoHarness = new THREE.Group();

  // 1. VBAT 7.4V 22AWG Red & Black: 2S BMS (-18,0,33) -> 6A Switch (-17,14,14.5) -> Mini560 (11,-21,25) & Mini-360 (11,21,23)
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

  // 2. 5.6V 5A Servo High-Current Rail: Mini560 (11,-21,28) -> 1000uF Bulk Cap (14,-14,17) -> PCA9685 V+ Screw Terminal (5,0,41)
  torsoHarness.add(createWireTube([
    [11.0, -20.0, 28.0], [13.8, -14.5, 19.5], [10.5, -8.0, 32.0], [5.8, -2.5, 41.5]
  ], 0.95, mats.wireServo5v6));
  torsoHarness.add(createWireTube([
    [11.0, -20.0, 26.0], [13.2, -13.0, 18.5], [9.5, -5.0, 32.0], [5.8, 2.5, 41.5]
  ], 0.90, mats.wireGndBlack));

  // 3. 5.0V Isolated Logic Rail: Mini-360 + 1N5819 (11,21,34) -> MAX98357A VIN (12,20,45) & Neck Umbilical (0,0,60)
  torsoHarness.add(createWireTube([
    [11.0, 20.0, 34.5], [12.0, 20.0, 42.0], [7.5, 11.0, 52.0], [-2.0, 2.0, 61.0]
  ], 0.72, mats.wireLogic5v0));

  // 4. 400 kHz I2C Bus (SDA=Blue, SCL=Amber): Neck Umbilical (-2,-2,61) -> PCA9685 I2C (3,30,33) -> Pelvis CoM MPU-6050 (-4,0,15)
  torsoHarness.add(createWireTube([
    [-2.5, -1.5, 61.0], [2.0, 18.0, 48.0], [3.0, 29.5, 34.0], [3.5, 14.0, 19.0], [-4.2, -1.2, 15.5]
  ], 0.62, mats.wireI2cSda));
  torsoHarness.add(createWireTube([
    [-2.5, -2.8, 61.0], [1.2, 17.0, 47.2], [3.0, 29.5, 31.5], [2.8, 13.0, 18.2], [-4.2, 1.2, 15.5]
  ], 0.62, mats.wireI2cScl));

  // 5. I2S Digital Audio Bus + Twisted Red/Black Speaker Leads -> 28mm Chest Speaker
  torsoHarness.add(createWireTube([
    [-1.5, 3.0, 60.5], [6.5, 14.0, 51.0], [11.5, 19.0, 44.5]
  ], 0.68, mats.wireI2sAudio));
  torsoHarness.add(createWireTube([
    [14.5, 18.0, 34.0], [17.2, 10.5, 29.0], [17.8, 5.5, 26.0]
  ], 0.62, mats.wireVbatRed));
  torsoHarness.add(createWireTube([
    [14.5, 16.5, 33.2], [17.0, 8.5, 28.0], [17.8, -5.5, 26.0]
  ], 0.62, mats.wireGndBlack));

  // 6. Head Internal I2C + 3.3V Flex Ribbon (XIAO ESP32S3 <-> Coaxial VL53L0X)
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

  // 7. Left & Right Leg Internal Conduit Servo Cables (routed inside leg_link.scad's X=-6.8mm rear bore!)
  function makeLegHarness() {
    const lg = new THREE.Group();
    lg.add(createWireTube([
      [-3.5, 0.0, -4.0], [-6.8, -0.9, -20.0], [-6.8, -0.9, -58.0], [-5.5, -0.8, -79.0]
    ], 0.68, mats.wireServo5v6, 20));
    lg.add(createWireTube([
      [-3.5, 1.2, -4.0], [-6.8, 0.9, -20.0], [-6.8, 0.9, -58.0], [-5.5, 0.8, -83.0]
    ], 0.62, mats.wirePwmSig, 20));
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
 *   - Red = Roll (+X), Green = Pitch (+Y), Blue = Yaw (+Z)
 *   - Golden CoM Sphere at Global [0.99, 0.0, 111.55] mm + Plumb Line onto Support Polygon!
 */
export function attachKinematicAndCoMGuides({
  torsoGroup, headYawJoint, headPitchJoint, lShoulderJoint, rShoulderJoint, leftLeg, rightLeg
}) {
  const markers = [];
  const matAxisX = new THREE.LineBasicMaterial({ color: 0xef4444, transparent: true, opacity: 0.78 });
  const matAxisY = new THREE.LineBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.78 });
  const matAxisZ = new THREE.LineBasicMaterial({ color: 0x2563eb, transparent: true, opacity: 0.78 });
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

  // 12 Joint Rotational Axes
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

  // Whole-Robot CoM Indicator in torsoGroup (Global Z = 111.55 mm => Torso local Z = 111.55 - 124.0 = -12.45 mm, X = +0.99 mm)
  const comGroup = new THREE.Group();
  comGroup.position.set(0.99 * MM, 0, -12.45 * MM);
  comGroup.visible = false;

  const comSphere = new THREE.Mesh(new THREE.SphereGeometry(3.4 * MM, 20, 20), matCoM);
  comGroup.add(comSphere);

  const comRing = new THREE.Mesh(new THREE.TorusGeometry(6.0 * MM, 0.45 * MM, 10, 32), matCoM);
  comGroup.add(comRing);

  // Plumb line down to Z=0 support polygon + support polygon outline!
  const plumbGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0, 0, -111.55 * MM),
  ]);
  const plumbLine = new THREE.Line(plumbGeo, new THREE.LineDashedMaterial({
    color: 0xd97706, dashSize: 0.005, gapSize: 0.003, transparent: true, opacity: 0.75
  }));
  plumbLine.computeLineDistances();
  comGroup.add(plumbLine);

  // Central flat-sole support polygon outline on floor (-25..+32mm X, -49..+49mm Y)
  const polyPts = [
    [-25.99, -49, -111.55], [31.01, -49, -111.55],
    [31.01,   49, -111.55], [-25.99, 49, -111.55], [-25.99, -49, -111.55],
  ].map((p) => new THREE.Vector3(p[0] * MM, p[1] * MM, p[2] * MM));
  const polyLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(polyPts),
    new THREE.LineBasicMaterial({ color: 0xd97706, transparent: true, opacity: 0.55 })
  );
  comGroup.add(polyLine);

  torsoGroup.add(comGroup);
  markers.push(comGroup);

  return markers;
}
