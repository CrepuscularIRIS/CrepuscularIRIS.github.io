import{P as p,C as t,d as f,M as h}from"./index-JUiUtJJc.js";import{S as m,n as w}from"./texlocal-CV06p3xw.js";import{SHORE as r}from"./shore-hnatMn7Q.js";import{BC_GLSL as g}from"./cloudlayer-CwKeZ8db.js";import"./react-three-fiber.esm-P4tynF3B.js";const x=`
  varying vec3 vWorld;
  void main(){
    vec4 w = modelMatrix * vec4(position, 1.0);
    vWorld = w.xyz;
    gl_Position = projectionMatrix * viewMatrix * w;
  }`,y=`
  uniform float uWT, uFogNear, uFogFar, uSpec, uGain, uFres, uWaveAmp;
  uniform vec3 uSunCol, uFog, uShallow, uNear, uFar;
  ${m}
  ${g}
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
    float s = max(dot(R, uSunDir), 0.0);
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
    vec3 foamCol = vec3(1.0, 0.97, 1.0) * mix(0.78, 1.05, uGain) * (1.0 - 0.18 * bcShadow(vec3(vWorld.x, 0.0, vWorld.z)));
    col = mix(col, foamCol, foam * 0.40);
    float f = smoothstep(uFogNear, uFogFar, dist);
    col = mix(col, uFog, f);
    /* shallow water is translucent: the wet sand shows through at the edge */
    float alpha = max(smoothstep(0.0, 2.2, d + (vn(fp * 1.3) - 0.5) * 0.5), foam * 0.7);
    gl_FragColor = vec4(col, alpha);
  }`;function C(n,e,z,c,i){const l=new p(1400,1400,2,2);l.rotateX(-Math.PI/2),l.translate(640,0,-500);const v=new t(e.seaShallow),u=new t(e.seaNear),d=new t(e.seaFar),s=new f({vertexShader:x,fragmentShader:y,transparent:!0,uniforms:{uShallow:{value:v},uNear:{value:u},uFar:{value:d},uSZ:{value:r.map(a=>a[0])},uSX:{value:r.map(a=>a[1])},...c,...i,uWT:{value:0},uFres:{value:e.seaFres??.55},uWaveAmp:{value:e.waveAmp??1},uFogNear:{value:e.fog.near},uFogFar:{value:e.fog.far*.9},uSpec:{value:e.seaSpec??.8},uGain:{value:e.seaGain??.5},uSunCol:{value:new t(e.glitter)},uFog:{value:new t(e.fog.color)}}}),o=w(new h(l,s));return o.name="sea",o.userData.dynamic=!0,o.frustumCulled=!1,n.add(o),{sea:o,update:a=>{s.uniforms.uWT.value=a*.3}}}export{C as buildSea};
