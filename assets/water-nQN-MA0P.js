import{P as S,z as o,x as F,F as m,V as b,I as G,e as y}from"./index-Cbw5Gy4V.js";import{$ as z,n as W}from"./texlocal-crzYRval.js";import{SHORE as f,SWASH_GLSL as k}from"./shore-DEkx_17n.js";import{BC_GLSL as C}from"./cloudlayer-DiINfybv.js";import"./react-three-fiber.esm-DdAgdf_9.js";import"./atmos-C7ngbJfe.js";import"./smoke-Z0TqTCHo.js";import"./motes-VBNxz0Vt.js";import"./uw-jjNVLKVY.js";const T=`
  varying vec3 vWorld;
  void main(){
    vec4 w = modelMatrix * vec4(position, 1.0);
    vWorld = w.xyz;
    gl_Position = projectionMatrix * viewMatrix * w;
  }`,p=`
  uniform float uWT, uFogNear, uFogFar, uSpec, uGain, uFres, uWaveAmp, uFoamGain;
  uniform vec3 uSunCol, uFog, uShallow, uNear, uFar, uShoreGlow, uFoamTint;
  ${z}
  ${C}
  uniform float uSZ[11]; uniform float uSX[11];
  varying vec3 vWorld;
  float shoreX(float z){
    float x = uSX[10];
    for (int i = 0; i < 10; i++) { float z0 = uSZ[i], z1 = uSZ[i+1]; if (z <= z0 && z >= z1) x = mix(uSX[i], uSX[i+1], (z0 - z) / (z0 - z1)); }
    return z > uSZ[0] ? uSX[0] : x;
  }
  float h21(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
  float vn(vec2 p){ vec2 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f);
    return mix(mix(h21(i), h21(i+vec2(1,0)), f.x), mix(h21(i+vec2(0,1)), h21(i+vec2(1,1)), f.x), f.y); }
  /* long, slow swell with crests running across the view: the sun glitter then breaks into a few stacked horizontal
     bars in a narrow column (as the plate paints it) instead of sparkle.  Wavelengths 6..25 m. */
  vec2 ripple(vec2 p, float t){
    float a = vn(vec2(p.x*0.12, p.y*0.34) + vec2(t*0.04, -t*0.05)), b = vn(vec2(p.x*0.12, p.y*0.34) + vec2(5.2, 1.3) - vec2(t*0.03, t*0.09));
    float c = vn(vec2(p.x*0.3, p.y*0.8) + vec2(-t*0.05, t*0.08)), d = vn(vec2(p.x*0.3, p.y*0.8) + vec2(9.1, 4.7) + vec2(t*0.04, t*0.10));
    vec2 s = vec2(0.85 * (a - b), 0.9 * (a - b) + 0.5 * (c - d));
    vec2 q = p * 0.42;
    s += 0.7 * cos(dot(q, vec2(0.16, 0.99)) * 2.1 - t * 0.5) * vec2(0.16, 0.99);
    s += 0.5 * cos(dot(q, vec2(-0.22, 0.98)) * 3.3 - t * 0.6) * vec2(-0.22, 0.98);
    s += 0.2 * cos(dot(q, vec2(0.35, 0.94)) * 5.1 - t * 0.7) * vec2(0.35, 0.94);
    return s;
  }
  /* lacy foam net: thin ridged lines of two noise scales leave small holes, like real sea foam */
  float lacy(vec2 p){
    vec2 w = vec2(vn(p * 0.8 + 3.1), vn(p * 0.8 + 8.7)) - 0.5;
    vec2 q = (p + w * 0.8) * vec2(1.7, 0.8);                      /* a little longer along the shore (z) */
    float blot = vn(q * 1.9) * 0.5 + vn(q * 4.3 + 4.1) * 0.3 + vn(q * 10.0 + 7.7) * 0.2;
    float net = 1.0 - abs(2.0 * (vn(q * 3.1 + 1.3) * 0.6 + vn(q * 7.3 + 6.1) * 0.4) - 1.0);
    return clamp(smoothstep(0.44, 0.64, blot) * 0.75 + smoothstep(0.88, 0.985, net) * 0.45, 0.0, 1.0);
  }
  float band(float d, float c, float w){ return 1.0 - smoothstep(w * 0.4, w, abs(d - c)); }
  void main(){
    vec3 toEye = cameraPosition - vWorld;
    float dist = length(toEye);
    vec3 V = toEye / dist;
    float d = vWorld.x - shoreX(vWorld.z);
    if (d < -0.05) discard;
    float near = 1.0 - smoothstep(30.0, 260.0, dist);
    vec2 r = ripple(vWorld.xz, uWT * 0.35); /* slow swell: fast ripples made the sun glints twinkle */
    /* near-field wave sets running parallel to the waterline: peaked crests that travel toward the shore, steepening
       as they near it (phase gets a slow along-shore wobble so the lines are not ruler straight) */
    float wph = d * 0.58 + uWT * 1.7 + 2.4 * vn(vec2(vWorld.z * 0.06, d * 0.04));
    float wv = 0.5 + 0.5 * sin(wph);
    float wcrest = pow(wv, 4.0);
    float wzone = smoothstep(0.3, 2.0, d) * (1.0 - smoothstep(16.0, 46.0, d)) * (1.0 - smoothstep(25.0, 90.0, dist));
    float wslope = cos(wph) * pow(wv, 3.0) * 2.0;
    vec3 n = normalize(vec3(r.x * 0.13 * (0.35 + near) + r.y * 0.07 - wslope * 0.34 * wzone * uWaveAmp, 1.0, r.y * 0.13 * (0.35 + near) + r.x * 0.04));
    vec3 R = reflect(-V, n);
    float cosv = clamp(dot(V, n), 0.0, 1.0);
    float F = 0.03 + 0.97 * pow(1.0 - cosv, 5.0);
    vec3 sky = skyColor(normalize(vec3(R.x, max(R.y, 0.015), R.z))); /* the real sky: clouds, sun glow, gradient */
    /* depth colour: teal shallows fading into the deep colour with distance from the waterline */
    vec3 deep = mix(uNear, uFar, smoothstep(20.0, 110.0, length(vWorld.xz)));
    vec3 base = mix(uShallow, deep, smoothstep(0.0, 5.0, d));
    /* Photon-like: dark body colour, the sky takes over toward the horizon (fresnel), a little at normal incidence */
    vec4 bcl = bcClouds(normalize(vec3(R.x, max(R.y, 0.015), R.z)));
    sky = mix(sky, bcl.rgb, bcl.a);                          /* the mackerel deck reflected too */
    vec3 col = mix(base * uGain, sky * 0.9, clamp(F * uFres + 0.02, 0.0, uFres));
    col *= 1.0 - 0.2 * bcShadow(vec3(vWorld.x, 0.0, vWorld.z)) * (1.0 - smoothstep(60.0, 300.0, dist));   /* slow cloud shadow */
    col = mix(col, base * uGain * 1.25 + vec3(0.02, 0.06, 0.06), wcrest * wzone * 0.28);   /* lit translucent wave faces */
    #ifdef BEACH_NIGHT
    float s = max(dot(R, uMoonDir), 0.0);
    #else
    float s = max(dot(R, uSunDir), 0.0);
    #endif
    /* round 9: narrower glint lobes, and no glitter in the shallows (the white blob over the wet sand came from here) */
    float glint = pow(s, 320.0) * 0.9 + pow(s, 90.0) * 0.14 + pow(s, 14.0) * 0.025;
    glint *= smoothstep(1.0, 9.0, d);
    col += uSunCol * glint * uSpec;
    col += uSunCol * 0.04 * (r.x + r.y);
    /* foam: slow surges that follow the real waterline, lacy, faded with distance; plus the breaking wave lines and
       the trailing foam they leave behind them */
    vec2 fp = vWorld.xz;
    float lace = lacy(fp * 1.6 + vec2(uWT * 0.05, 0.0));
    float lace2 = lacy(fp * vec2(2.4, 1.1) + vec2(3.7, uWT * 0.02));
    float surge = 0.5 + 0.5 * sin(uWT * 0.42 + vWorld.z * 0.05);
    float surge2 = 0.5 + 0.5 * sin(uWT * 0.31 + vWorld.z * 0.07 + 1.7);
    float foam = band(d, 0.12 + 0.55 * surge, 0.34) * (0.35 + 0.65 * lace)
               + band(d, 1.5 + 0.9 * surge2, 0.42) * lace * 0.9
               + band(d, 3.4 + 0.7 * sin(uWT * 0.27 + vWorld.z * 0.09), 0.5) * lace * 0.5
               + (1.0 - smoothstep(0.0, 0.5 + 0.4 * surge, d)) * 0.55;
    float brk = smoothstep(0.55, 0.92, wcrest) * (1.0 - smoothstep(5.0, 17.0, d)) * smoothstep(0.6, 1.6, d);
    float trail = pow(0.5 + 0.5 * sin(wph - 1.15), 5.0) * (1.0 - smoothstep(4.0, 12.0, d)) * smoothstep(0.5, 1.5, d);
    foam += brk * (0.35 + 0.8 * lace2) + trail * lace * 0.65;
    foam *= (1.0 - smoothstep(18.0, 110.0, dist)) * (0.5 + 0.5 * lace);
    foam = clamp(foam, 0.0, 1.0);
    #ifdef BEACH_NIGHT
    vec3 foamCol = uFoamTint * mix(0.78, 1.05, uGain) * uFoamGain * (1.0 - 0.18 * bcShadow(vec3(vWorld.x, 0.0, vWorld.z)));
    #else
    vec3 foamCol = vec3(1.0, 0.97, 1.0) * mix(0.78, 1.05, uGain) * (1.0 - 0.18 * bcShadow(vec3(vWorld.x, 0.0, vWorld.z)));
    #endif
    col = mix(col, foamCol, foam * 0.40);
    #ifdef BEACH_NIGHT
    /* Snowdrop's glow stays in the existing breaking-wave and foam masks: no emissive open sea, dry sand or new noise. */
    float glowMask = clamp(foam * 0.70 + brk * 0.35 + trail * 0.25, 0.0, 1.0)
                   * (1.0 - smoothstep(1.5, 8.0, d)) * (1.0 - smoothstep(18.0, 60.0, dist));
    col += uShoreGlow * glowMask;
    #endif
    float f = smoothstep(uFogNear, uFogFar, dist);
    col = mix(col, uFog, f);
    /* shallow water is translucent: the wet sand shows through at the edge */
    float alpha = max(smoothstep(0.0, 2.2, d + (vn(fp * 1.3) - 0.5) * 0.5), foam * 0.7);
    gl_FragColor = vec4(col, alpha);
  }`,u=p.replace("uniform vec3 uSunCol, uFog, uShallow, uNear, uFar, uShoreGlow, uFoamTint;",`uniform vec3 uSunCol, uFog, uShallow, uNear, uFar, uShoreGlow, uFoamTint, uGlK, uGlW, uLampP, uLampC;
  uniform float uGlNear, uLampW;`).replace(`float glint = pow(s, 320.0) * 0.9 + pow(s, 90.0) * 0.14 + pow(s, 14.0) * 0.025;
    glint *= smoothstep(1.0, 9.0, d);`,`
    /* a column under the moon: narrow across (azimuth), long down the sea (elevation), broken by the swell */
    float az = 1.0 - dot(normalize(R.xz + 1e-4), normalize(uMoonDir.xz)), el = R.y - uMoonDir.y;
    float path = exp(-az * uGlK.y - el * el * uGlK.z) * smoothstep(0.15, 0.85, 0.5 + 0.5 * (r.x + r.y));
    float glint = pow(s, uGlK.x) * uGlW.x + path * uGlW.y;
    glint *= smoothstep(1.0, uGlNear, d);`).replace("col += uShoreGlow * glowMask;",`col += uShoreGlow * glowMask;
    /* the lit buoy's streak toward the eye (core/water.js lamp streaks), broken by the swell */
    vec2 vd = normalize(V.xz + 1e-4), off = vWorld.xz - uLampP.xz;
    float along = dot(off, vd), lat = dot(off, vec2(-vd.y, vd.x)), lw = uLampW * (1.0 + 0.15 * r.x);
    float streak = exp(-lat * lat / (lw * lw)) * exp(-max(along, 0.0) * 0.16) * smoothstep(-1.0, 0.5, along);
    col += uLampC * streak * smoothstep(0.25, 0.75, vn(vec2(lat * 3.0, along * 1.8 + uWT * 0.3))) * 1.6 * (1.0 - cosv * 0.6);`);if(u.split("uGlK").length<3||!u.includes("uLampC * streak"))throw new Error("Beach water: night splice did not apply");const l=p.replace("uniform float uWT, uFogNear, uFogFar, uSpec, uGain, uFres, uWaveAmp, uFoamGain;","uniform float uWT, uFogNear, uFogFar, uSpec, uGain, uFres, uWaveAmp, uFoamGain, uFoamScale;").replace("vec2 r = ripple(vWorld.xz, uWT * 0.35);","vec2 r = ripple(vWorld.xz, uWT * 0.35) * uWaveAmp;").replace("float lace = lacy(fp * 1.6 +","float lace = lacy(fp * 1.6 * uFoamScale +").replace("float lace2 = lacy(fp * vec2(2.4, 1.1) +","float lace2 = lacy(fp * vec2(2.4, 1.1) * uFoamScale +").replace("vec3 foamCol = vec3(1.0, 0.97, 1.0) * mix(0.78, 1.05, uGain) *","vec3 foamCol = vec3(1.0, 0.97, 1.0) * mix(0.78, 1.05, uGain) * uFoamGain *").replace("smoothstep(0.44, 0.64, blot) * 0.75","smoothstep(0.52, 0.70, blot) * 0.5");if(l.split("uFoamScale").length<4||!l.includes("* uWaveAmp;")||!l.includes("uGain) * uFoamGain *")||!l.includes("0.70, blot"))throw new Error("Beach water: dusk splice did not apply");const N=(a,e,r)=>{if(a.split(e).length!==2)throw new Error(`Beach water: splice anchor not unique: ${e.slice(0,48)}`);return a.replace(e,r)},h=a=>[["uniform float uSZ[11];",`${k}
  uniform vec2 uClear;
  uniform float uSZ[11];`],["vec3 col = mix(base * uGain, sky * 0.9, clamp(F * uFres + 0.02, 0.0, uFres));",`vec3 col = mix(base * uGain, sky * 0.9, clamp(F * uFres + 0.02, 0.0, uFres));
    float bw = 1.0 - clamp(F * uFres + 0.02, 0.0, uFres);   /* the body colour's share of col */`],["/* slow cloud shadow */",`/* slow cloud shadow */
    bw *= 1.0 - 0.2 * bcShadow(vec3(vWorld.x, 0.0, vWorld.z)) * (1.0 - smoothstep(60.0, 300.0, dist));`],["wcrest * wzone * 0.28);",`wcrest * wzone * 0.28);
    bw = mix(bw, 1.25, wcrest * wzone * 0.28);`],["float foam = band(d, 0.12 + 0.55 * surge, 0.34) * (0.35 + 0.65 * lace)",`vec2 swSh; vec4 sw = swash(-d, vWorld.z, uTime, swSh);
    float foam = (sw.x + sw.y * (1.0 - smoothstep(0.5, 2.5, d))) * (0.35 + 0.65 * lace)`],["+ (1.0 - smoothstep(0.0, 0.5 + 0.4 * surge, d)) * 0.55;",";"],["col = mix(col, foamCol, foam * 0.40);",`col = mix(col, foamCol, foam * 0.40);
    bw *= 1.0 - foam * 0.40;`],["col = mix(col, uFog, f);",`col = mix(col, uFog, f);
    bw *= 1.0 - f;`],[`float alpha = max(smoothstep(0.0, 2.2, d + (vn(fp * 1.3) - 0.5) * 0.5), foam * 0.7);
    gl_FragColor = vec4(col, alpha);`,`
    float tr = (1.0 - smoothstep(uClear.x, uClear.y, d + (vn(fp * 1.3) - 0.5) * 0.5)) * smoothstep(36.0, 32.0, vWorld.z) * smoothstep(-216.0, -212.0, vWorld.z);
    float bt = clamp(bw * tr, 0.0, 1.0);
    float bare = smoothstep(0.02, 0.35, -d - swSh.y);   /* the backwash has drained off this sand */
    gl_FragColor = vec4(max(col - base * uGain * bt, 0.0), 1.0 - bt) * (1.0 - bare);`]].reduce((e,[r,c])=>N(e,r,c),a),L=h(u),R=h(l);function B(a,e,r,c,d){const i=new S(1400,1400,2,2);i.rotateX(-Math.PI/2),i.translate(640,0,-500);const w=new o(e.seaShallow),g=new o(e.seaNear),x=new o(e.seaFar),n=!!e.swash,v=new F({vertexShader:T,fragmentShader:e.night?n?L:u:e.foamScale?n?R:l:p,transparent:!0,...n?{premultipliedAlpha:!0}:{},...e.night?{defines:{BEACH_NIGHT:1}}:{},uniforms:{uShallow:{value:w},uNear:{value:g},uFar:{value:x},uSZ:{value:f.map(s=>s[0])},uSX:{value:f.map(s=>s[1])},...c,...d,uWT:{value:0},uFres:{value:e.seaFres??.55},uWaveAmp:{value:e.waveAmp??1},uFogNear:{value:e.fog.near},uFogFar:{value:e.fog.far*.9},uSpec:{value:e.seaSpec??.8},uGain:{value:e.seaGain??.5},uSunCol:{value:new o(e.glitter)},uFog:{value:new o(e.fog.color)},uShoreGlow:{value:new o(e.shoreGlow??0).multiplyScalar(e.shoreGlowGain??1)},uFoamTint:{value:new o(e.foam??16777215)},uFoamGain:{value:e.foamGain??1},...e.foamScale?{uFoamScale:{value:e.foamScale}}:{},...n?{uSwash:{value:new G(...e.swash)},uClear:{value:new b(...e.waterClear)}}:{},...e.night?{uGlK:{value:new m(...e.moonGlint.k)},uGlW:{value:new m(...e.moonGlint.w)},uGlNear:{value:e.moonGlint.near},uLampP:{value:new m(...e.seaLamp.at)},uLampC:{value:new o(e.seaLamp.color).multiplyScalar(e.seaLamp.gain)},uLampW:{value:e.seaLamp.w}}:{}}}),t=W(new y(i,v));return t.name="sea",t.userData.dynamic=!0,t.frustumCulled=!1,a.add(t),{sea:t,update:s=>{v.uniforms.uWT.value=s*.3}}}export{B as buildSea};
