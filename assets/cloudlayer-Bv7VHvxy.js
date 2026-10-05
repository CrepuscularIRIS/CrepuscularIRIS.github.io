import{C as c,d as n,s as l,M as u,j as d}from"./index-LnGnRg_2.js";import{S as f,L as p}from"./texlocal-CsbVczrI.js";import"./react-three-fiber.esm-CDfrq34R.js";const x=o=>({uBcLit:{value:new c(o.bc.lit)},uBcShade:{value:new c(o.bc.shade)},uBcWarm:{value:new c(o.bc.warm)},uBcCover:{value:o.bc.cover},uBcAlpha:{value:o.bc.alpha}}),g=`
  uniform float uTime; uniform vec3 uSunDir;
`,v=`
  uniform vec3 uBcLit, uBcShade, uBcWarm;
  uniform float uBcCover, uBcAlpha;
  /* own noise with a sin-free hash: sin() of large arguments loses precision on real GPUs and the clouds turn into
     blocky stair-steps (the core sky's hash does that) */
  float bcH(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
  float skyNoiseB(vec2 p){
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(bcH(i), bcH(i + vec2(1, 0)), f.x), mix(bcH(i + vec2(0, 1)), bcH(i + vec2(1, 1)), f.x), f.y);
  }
  float skyFbmB(vec2 p){ float a = 0.5, s = 0.0; for (int i = 0; i < 5; i++) { s += a * skyNoiseB(p); p = p * 2.03 + vec2(17.1, 9.2); a *= 0.5; } return s; }
  float bcDens(vec2 p){
    float w = skyFbmB(p * 0.3 + 2.0);
    vec2 q = vec2(p.x + w * 1.6, p.y);
    float rows = 0.5 + 0.5 * sin(q.y * 3.3 + w * 5.0);
    float lump = skyNoiseB(vec2(q.x * 2.6, q.y * 3.8) + w * 2.2);
    float fine = skyFbmB(q * 5.2 + 11.0);
    float d = rows * 0.5 + lump * 0.7 + fine * 0.4 - 0.66 + uBcCover;
    return smoothstep(0.0, 0.36, d);
  }
  /* colour (rgb) and coverage (a) of the deck seen along direction d */
  vec4 bcClouds(vec3 d){
    float h = d.y;
    if (h < 0.05) return vec4(0.0);
    vec2 p = d.xz / (h + 0.16) * 2.3 + vec2(uTime * 0.004, uTime * 0.0016);
    float dens = bcDens(p);
    if (dens < 0.003) return vec4(0.0);
    vec2 ts = normalize(uSunDir.xz + 1e-4) * 0.13;
    float sh = (bcDens(p + ts) * 0.6 + bcDens(p + ts * 2.4) * 0.4);
    float lit = clamp(1.15 - sh * 1.15, 0.0, 1.0);
    float sd = max(dot(d, uSunDir), 0.0);
    vec3 c = mix(uBcShade, uBcLit, lit);
    c += uBcWarm * (pow(sd, 8.0) * 0.28 + pow(sd, 60.0) * 0.3) * (0.35 + lit);
    c = mix(c, c * 0.8 + uBcShade * 0.25, (1.0 - dens) * 0.3);                 /* thin edges slightly bluer */
    float fade = smoothstep(0.05, 0.3, h);
    return vec4(c, dens * fade * uBcAlpha);
  }
  /* soft shadow of the deck on the ground: 0 clear, 1 under a cloud.  Sampled along the key-light direction at a
     low pretend altitude so the pattern is a few tens of metres wide and drifts slowly. */
  float bcShadow(vec3 g){
    const vec3 L = vec3(-0.15, 0.93, 0.40);
    const float A = 240.0;
    vec3 c = g + L * (A / L.y);
    float len = length(vec3(c.x, A, c.z));
    vec2 p = c.xz / (A + 0.16 * len) * 2.3 + vec2(uTime * 0.004, uTime * 0.0016);
    return smoothstep(0.1, 0.9, bcDens(p));
  }
`,m=`
  varying vec3 vDir;
  void main(){
    vDir = normalize(position);
    vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_Position = p.xyww;
  }`;function S(o,t,a){const r={...t,...a},s=new n({uniforms:r,vertexShader:m,side:l,transparent:!0,depthWrite:!1,depthTest:!0,fog:!1,fragmentShader:`
      varying vec3 vDir;
      ${f}
      ${v}
      void main(){
        vec4 c = bcClouds(normalize(vDir));
        if (c.a < 0.002) discard;
        gl_FragColor = vec4(c.rgb, c.a);
      }`}),e=new u(new d(1,48,24),s);return e.name="cloudDeck",e.renderOrder=-999,e.frustumCulled=!1,e.layers.set(p),e.scale.setScalar(590),e.onBeforeRender=(h,y,i)=>e.position.copy(i.position),e.userData.dynamic=!0,o.add(e),e}export{v as BC_GLSL,g as BC_NOISE,x as bcUniforms,S as buildCloudLayer};
