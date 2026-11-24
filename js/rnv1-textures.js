// RNV1 High-DPI Procedural PCB Copper-Trace, Silkscreen, IC Laser-Etch,
// Li-Ion Cell Wrapper & Engineering Cross-Section Hatch Shader Generator.
// Zero external image requests — 100% crisp procedural vector-to-CanvasTexture synthesis.
import * as THREE from '../vendor/three/three.module.min.js';

function makeCanvasTex(w, h, drawFn, maxAnisotropy = 8) {
  const cv = document.createElement('canvas');
  cv.width = w;
  cv.height = h;
  const ctx = cv.getContext('2d');
  drawFn(ctx, w, h);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = maxAnisotropy;
  tex.needsUpdate = true;
  return tex;
}

function drawViaGrid(ctx, x0, y0, cols, rows, dx, dy, rOuter = 3.2, rHole = 1.4) {
  for (let r = 0; r < rows; ++r) {
    for (let c = 0; c < cols; ++c) {
      const x = x0 + c * dx;
      const y = y0 + r * dy;
      ctx.fillStyle = '#eab308';
      ctx.beginPath(); ctx.arc(x, y, rOuter, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#0b1017';
      ctx.beginPath(); ctx.arc(x, y, rHole, 0, Math.PI * 2); ctx.fill();
    }
  }
}

export function createMicroComponentTextures(maxAnisotropy = 8) {
  // 1. PCA9685 16-Channel PWM Driver PCB (Royal Blue Solder Mask + Copper Traces + Silkscreen)
  const pca9685Tex = makeCanvasTex(512, 256, (ctx, w, h) => {
    ctx.fillStyle = '#0f3a6d';
    ctx.fillRect(0, 0, w, h);

    // Subtle etched copper ground plane + 16 PWM signal trace buses from central TSSOP-28
    ctx.fillStyle = '#164b8a';
    ctx.fillRect(14, 14, w - 28, h - 28);

    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 2.2;
    for (let ch = 0; ch < 16; ++ch) {
      const tx = 54 + ch * 26.8;
      ctx.beginPath();
      ctx.moveTo(w * 0.5 + (ch - 7.5) * 4.2, h * 0.40);
      ctx.lineTo(tx, h * 0.56);
      ctx.lineTo(tx, h * 0.84);
      ctx.stroke();
    }

    // Thick 5.6V V+ high-current copper bus trace
    ctx.strokeStyle = '#1d5aa3';
    ctx.lineWidth = 10;
    ctx.beginPath();
    ctx.moveTo(w * 0.5, 36);
    ctx.lineTo(w * 0.5, h * 0.74);
    ctx.moveTo(42, h * 0.74);
    ctx.lineTo(w - 42, h * 0.74);
    ctx.stroke();

    drawViaGrid(ctx, 195, 78, 7, 3, 18, 12, 2.4, 1.0);

    // Crisp white PCB silkscreen
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.lineWidth = 2;
    ctx.strokeRect(8, 8, w - 16, h - 16);
    ctx.strokeRect(w * 0.5 - 38, 68, 76, 44); // TSSOP-28 courtyard

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 15px ui-monospace, monospace';
    ctx.fillText('PCA9685 16-CH 12-BIT PWM', 32, 34);
    ctx.font = 'bold 13px ui-monospace, monospace';
    ctx.fillText('I2C:0x40  OE#:D9', 332, 34);
    ctx.fillText('V+ 5.6V 5A', 216, 56);

    ctx.font = 'bold 11px ui-monospace, monospace';
    ctx.fillStyle = '#fde047';
    ctx.fillText('PWM', 14, 158);
    ctx.fillStyle = '#fca5a5';
    ctx.fillText('V+', 16, 184);
    ctx.fillStyle = '#cbd5e1';
    ctx.fillText('GND', 14, 210);

    ctx.fillStyle = '#ffffff';
    for (let ch = 0; ch < 16; ++ch) {
      const tx = 48 + ch * 26.8;
      ctx.fillText(String(ch), tx, 142);
    }
  }, maxAnisotropy);

  // 2. MPU-6050 (GY-521) CoM IMU PCB (Blue Mask + X/Y Axis Triad Silkscreen)
  const mpu6050Tex = makeCanvasTex(256, 256, (ctx, w, h) => {
    ctx.fillStyle = '#113e75';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#184e8f';
    ctx.fillRect(14, 14, w - 28, h - 28);

    drawViaGrid(ctx, 96, 96, 4, 4, 20, 20, 2.6, 1.1);

    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2.5;
    ctx.strokeRect(8, 8, w - 16, h - 16);
    ctx.strokeRect(98, 98, 60, 60); // QFN-24 outline

    // X / Y Inertial Reference Silkscreen Arrows
    ctx.beginPath();
    ctx.moveTo(128, 84); ctx.lineTo(128, 38);
    ctx.lineTo(120, 48); ctx.moveTo(128, 38); ctx.lineTo(136, 48);
    ctx.moveTo(172, 128); ctx.lineTo(218, 128);
    ctx.lineTo(208, 120); ctx.moveTo(218, 128); ctx.lineTo(208, 136);
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 18px ui-monospace, monospace';
    ctx.fillText('MPU-6050', 22, 232);
    ctx.font = 'bold 14px ui-monospace, monospace';
    ctx.fillText('+X', 140, 50);
    ctx.fillText('+Y', 198, 114);
    ctx.fillText('CoM 0x68', 148, 232);
  }, maxAnisotropy);

  // 3. Seeed Studio XIAO ESP32S3 Matte-Black PCB (Gold Pads + Pin Silkscreen)
  const xiaoPcbTex = makeCanvasTex(256, 256, (ctx, w, h) => {
    ctx.fillStyle = '#14181f';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = '#28303d';
    ctx.lineWidth = 2.2;
    for (let i = 0; i < 9; ++i) {
      ctx.beginPath();
      ctx.moveTo(30, 32 + i * 24);
      ctx.lineTo(w - 30, 32 + i * 24);
      ctx.stroke();
    }
    drawViaGrid(ctx, 76, 76, 5, 5, 24, 24, 2.5, 1.0);
    ctx.strokeStyle = 'rgba(255,255,255,0.75)';
    ctx.strokeRect(10, 10, w - 20, h - 20);
    ctx.fillStyle = '#f8fafc';
    ctx.font = 'bold 15px ui-monospace, monospace';
    ctx.fillText('XIAO ESP32S3', 62, 30);
    const lPins = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6'];
    const rPins = ['5V', 'GD', '3V', '10', 'D9', 'D8', 'D7'];
    ctx.font = 'bold 13px ui-monospace, monospace';
    for (let i = 0; i < 7; ++i) {
      const py = 58 + i * 28;
      ctx.fillStyle = '#eab308';
      ctx.fillRect(4, py - 9, 14, 14);
      ctx.fillRect(w - 18, py - 9, 14, 14);
      ctx.fillStyle = '#e2e8f0';
      ctx.fillText(lPins[i], 22, py + 3);
      ctx.fillText(rPins[i], w - 44, py + 3);
    }
  }, maxAnisotropy);

  // 4. Mini560 / Mini-360 High-Current Green Power Buck PCB (Wide Copper Pours + Thermal Vias)
  const buckPcbTex = makeCanvasTex(256, 256, (ctx, w, h) => {
    ctx.fillStyle = '#0f5330';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#1b7245';
    ctx.fillRect(14, 14, 104, h - 28);
    ctx.fillRect(138, 14, 104, h - 28);
    drawViaGrid(ctx, 28, 36, 4, 7, 22, 26, 3.0, 1.3);
    drawViaGrid(ctx, 152, 36, 4, 7, 22, 26, 3.0, 1.3);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2.2;
    ctx.strokeRect(8, 8, w - 16, h - 16);
    ctx.fillStyle = '#eab308';
    ctx.fillRect(18, 18, 42, 26);
    ctx.fillRect(w - 60, 18, 42, 26);
    ctx.fillRect(18, h - 44, 42, 26);
    ctx.fillRect(w - 60, h - 44, 42, 26);
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 16px ui-monospace, monospace';
    ctx.fillText('IN+', 22, 62);
    ctx.fillText('IN-', w - 56, 62);
    ctx.fillText('VO+', 22, h - 52);
    ctx.fillText('GND', w - 58, h - 52);
    ctx.fillText('SYNC BUCK', 80, 134);
  }, maxAnisotropy);

  // 5. 2S 5A BMS Protection PCB (Nickel Pad Legend: B- / BM 3.7V / B+ 7.4V)
  const bmsPcbTex = makeCanvasTex(256, 128, (ctx, w, h) => {
    ctx.fillStyle = '#115834';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#cbd5e1';
    ctx.fillRect(12, 12, 36, 32);
    ctx.fillRect(110, 12, 36, 32);
    ctx.fillRect(w - 48, 12, 36, 32);
    drawViaGrid(ctx, 64, 68, 6, 2, 24, 22, 2.6, 1.1);
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 16px ui-monospace, monospace';
    ctx.fillText('B-', 18, 62);
    ctx.fillText('BM 3.7V', 94, 62);
    ctx.fillText('B+ 7.4V', w - 78, 62);
    ctx.fillText('2S 5A Li-ion BMS', 52, 114);
  }, maxAnisotropy);

  // 6. Purple Sensor / Audio Breakout PCB (VL53L0X & MAX98357A)
  const purplePcbTex = makeCanvasTex(256, 256, (ctx, w, h) => {
    ctx.fillStyle = '#431e66';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = '#6b339e';
    ctx.lineWidth = 3;
    for (let i = 0; i < 7; ++i) {
      ctx.beginPath();
      ctx.moveTo(24 + i * 32, 24);
      ctx.lineTo(128, 128);
      ctx.lineTo(24 + i * 32, h - 24);
      ctx.stroke();
    }
    drawViaGrid(ctx, 88, 88, 4, 4, 26, 26, 2.5, 1.0);
    ctx.strokeStyle = 'rgba(255,255,255,0.82)';
    ctx.lineWidth = 2.2;
    ctx.strokeRect(8, 8, w - 16, h - 16);
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 16px ui-monospace, monospace';
    ctx.fillText('I2S / ToF OPTIC', 52, 34);
    ctx.fillText('3.3V / 5.0V', 72, 234);
  }, maxAnisotropy);

  // 7. 14500 800mAh 3.7V 10C High-Drain Li-Ion Cylindrical Cell Wrap
  const liIonCellTex = makeCanvasTex(512, 256, (ctx, w, h) => {
    const grad = ctx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0.0, '#154482');
    grad.addColorStop(0.5, '#1e60b5');
    grad.addColorStop(1.0, '#154482');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    // Industrial metallic band stripes at top & bottom
    ctx.fillStyle = '#eab308';
    ctx.fillRect(0, 16, w, 8);
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, h - 24, w, 8);

    ctx.fillStyle = 'rgba(255,255,255,0.92)';
    ctx.font = 'bold 24px ui-monospace, monospace';
    ctx.fillText('INR14500  800mAh  3.7V', 44, 96);
    ctx.font = 'bold 19px ui-monospace, monospace';
    ctx.fillStyle = '#bae6fd';
    ctx.fillText('2.96Wh • 10C HIGH-DRAIN Li-ion', 44, 134);
    ctx.fillStyle = '#fde047';
    ctx.font = 'bold 26px ui-monospace, monospace';
    ctx.fillText('[+]            [-]', 44, 188);
  }, maxAnisotropy);

  // 8. 1000uF 10V Low-ESR 105°C Radial Electrolytic Capacitor Sleeve
  const capSleeveTex = makeCanvasTex(256, 256, (ctx, w, h) => {
    ctx.fillStyle = '#13204a';
    ctx.fillRect(0, 0, w, h);
    // Silver negative polarity stripe with minus chevrons
    ctx.fillStyle = '#cbd5e1';
    ctx.fillRect(28, 0, 44, h);
    ctx.fillStyle = '#0f172a';
    ctx.font = 'bold 26px monospace';
    for (let y = 36; y < h; y += 52) {
      ctx.fillText('—', 37, y);
    }
    ctx.fillStyle = '#f8fafc';
    ctx.font = 'bold 20px ui-monospace, monospace';
    ctx.fillText('1000µF', 92, 92);
    ctx.fillText('10V 105°C', 92, 132);
    ctx.fillStyle = '#fde047';
    ctx.font = 'bold 16px ui-monospace, monospace';
    ctx.fillText('LOW-ESR', 92, 168);
  }, maxAnisotropy);

  // 9. Shielded Molded Alloy Inductor Laser Marking ("4R7")
  const inductor4R7Tex = makeCanvasTex(128, 128, (ctx, w, h) => {
    ctx.fillStyle = '#252932';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = '#3b4252';
    ctx.lineWidth = 4;
    ctx.strokeRect(8, 8, w - 16, h - 16);
    ctx.fillStyle = '#cbd5e1';
    ctx.font = 'bold 34px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('4R7', w * 0.5, h * 0.52);
  }, maxAnisotropy);

  // 10. Silicon IC Laser Package Marking ("ESP32-S3R8" / QFN)
  const icLaserTex = makeCanvasTex(128, 128, (ctx, w, h) => {
    ctx.fillStyle = '#15181e';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#eab308';
    ctx.beginPath(); ctx.arc(22, 22, 6, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = 'rgba(226,232,240,0.82)';
    ctx.font = 'bold 16px ui-monospace, monospace';
    ctx.fillText('RNV1-IC', 22, 62);
    ctx.font = '13px ui-monospace, monospace';
    ctx.fillText('240MHz S3', 22, 86);
  }, maxAnisotropy);

  return {
    pca9685Tex,
    mpu6050Tex,
    xiaoPcbTex,
    buckPcbTex,
    bmsPcbTex,
    purplePcbTex,
    liIonCellTex,
    capSleeveTex,
    inductor4R7Tex,
    icLaserTex,
  };
}

/**
 * Inject crisp 45-degree Engineering CAD Cross-Section Hatch Shader on watertight STL backfaces!
 * Because all 12 OpenSCAD STLs are 100% watertight manifolds, `!gl_FrontFacing` is visible ONLY
 * where a section-cutaway plane slices directly through the solid PETG/PLA+ wall thickness!
 */
export function applySectionCapHatchShader(material, hatchUniform, baseRgb = [0.80, 0.86, 0.93], lineRgb = [0.07, 0.42, 0.68]) {
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uSectionHatchActive = hatchUniform;
    shader.fragmentShader = `uniform float uSectionHatchActive;\n` + shader.fragmentShader;
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <dithering_fragment>',
      `
      if (!gl_FrontFacing && uSectionHatchActive > 0.5) {
        float diag = fract((gl_FragCoord.x + gl_FragCoord.y) * 0.14);
        float stripe = step(0.72, diag);
        vec3 capCol = mix(vec3(${baseRgb[0]}, ${baseRgb[1]}, ${baseRgb[2]}), vec3(${lineRgb[0]}, ${lineRgb[1]}, ${lineRgb[2]}), stripe);
        gl_FragColor = vec4(capCol, gl_FragColor.a);
      }
      #include <dithering_fragment>
      `
    );
  };
  material.customProgramCacheKey = () => `rnv1_section_hatch_${baseRgb[0].toFixed(2)}`;
}
