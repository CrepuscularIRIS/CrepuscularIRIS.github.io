import{aB as te,aC as ae,e as le,P as ue,x as S,aD as q,F as h,aE as oe,z as k,y as ie,V as R,I as d,W as se,aF as Y,aG as re,H as ne,D as ve}from"./index-Ce7HOOEM.js";import{L as fe}from"./texlocal-L9BQ2kdi.js";import"./react-three-fiber.esm-CXnyw8FP.js";const L=`
varying vec2 vUv;
void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,ce={enabled:!0,outline:{enabled:!0,color:1841710,opacity:.7,width:1,depthTh:.012,normalTh:.22},ao:{enabled:!0,strength:.45,radius:.35},bloom:{enabled:!0,threshold:.85,knee:.35,strength:.55,weights:[.5,.35,.3,0,0]},fog:{enabled:!1,color:8425640,density:.02,falloff:.4,base:0,near:0,maxOpacity:.9},rays:{enabled:!1,lights:[],threshold:.8,knee:.4,decay:.96,density:.9,samples:24},vol:{enabled:!1,color:null,strength:1,density:.05,max:.6,phase:.5,steps:24,dist:40,bias:.002,top:1e4,bottom:-1e4,floor:0},paint:{enabled:!1,strength:.6,radius:2},grade:{exposure:1,contrast:1,saturation:1,lift:[0,0,0],gain:[1,1,1],gamma:1,vignette:.25,grain:0,staticGrain:0,guard:0},dof:{enabled:!1,focus:4,range:2,near:1,far:1,maxRadius:6,tilt:{center:[.5,.5],radius:[.5,.5],soft:.25,strength:0}}};function Z(i,t){const a=Array.isArray(i)?i.slice():{...i};if(!t)return a;for(const u of Object.keys(t)){const e=t[u];e&&typeof e=="object"&&!Array.isArray(e)&&typeof i[u]=="object"&&!Array.isArray(i[u])?a[u]=Z(i[u],e):a[u]=e}return a}function J(i){return Z(ce,i||{})}function b(i,t,a={}){const u=new se(i,t,{type:ne,format:re,minFilter:Y,magFilter:Y,depthBuffer:a.depth!==!1,stencilBuffer:!1,generateMipmaps:!1});return a.samples&&(u.samples=a.samples),u}const he=`
  #include <common>
  #include <uv_pars_vertex>
  varying vec3 vN; varying float vD;
  void main(){
    #include <uv_vertex>
    #include <beginnormal_vertex>
    #include <defaultnormal_vertex>
    vN = normalize(transformedNormal);
    #include <begin_vertex>
    #include <project_vertex>
    vD = -mvPosition.z;
  }`,de=`
  varying vec3 vN; varying float vD;
  #ifdef ND_ALPHA
    uniform sampler2D uMap; uniform float uAlphaTest; varying vec2 vMapUv;
  #endif
  void main(){
    #ifdef ND_ALPHA
      if (texture2D(uMap, vMapUv).a < uAlphaTest) discard;
    #endif
    vec3 n = normalize(vN);
    if (!gl_FrontFacing) n = -n;
    gl_FragColor = vec4(n * 0.5 + 0.5, 1.0 / max(vD, 1e-3));
  }`,Q=new Map;function X(i){const t=i.alphaTest>0&&(i.map||i.alphaMap),a=t?i.map||i.alphaMap:null,u=`${i.side}|${t?a.id+":"+i.alphaTest:""}|${i.polygonOffset}|${i.polygonOffsetFactor}|${i.polygonOffsetUnits}|${i.depthTest}|${i.depthWrite}`;let e=Q.get(u);return e||(e=new S({vertexShader:he,fragmentShader:de,side:i.side,polygonOffset:i.polygonOffset,polygonOffsetFactor:i.polygonOffsetFactor,polygonOffsetUnits:i.polygonOffsetUnits,depthTest:i.depthTest,depthWrite:i.depthWrite,defines:t?{ND_ALPHA:"",USE_MAP:"",MAP_UV:"uv"}:{},uniforms:t?{uMap:{value:a},uAlphaTest:{value:i.alphaTest},mapTransform:{value:a.matrix}}:{}}),Q.set(u,e)),e}const _=new S({uniforms:{tDiffuse:{value:null},uTh:{value:.85},uKnee:{value:.35}},vertexShader:L,fragmentShader:`
    varying vec2 vUv; uniform sampler2D tDiffuse; uniform float uTh, uKnee;
    void main(){
      vec3 c = texture2D(tDiffuse, vUv).rgb;
      float l = max(max(c.r, c.g), c.b);
      float k = smoothstep(uTh, uTh + uKnee, l);
      gl_FragColor = vec4(min(c * k, vec3(16.0)), 1.0);
    }`,depthTest:!1,depthWrite:!1}),f=new S({uniforms:{tDiffuse:{value:null},uDir:{value:new R(1,0)}},vertexShader:L,fragmentShader:`
    varying vec2 vUv; uniform sampler2D tDiffuse; uniform vec2 uDir;
    void main(){
      vec3 s = texture2D(tDiffuse, vUv).rgb * 0.2270270;
      s += texture2D(tDiffuse, vUv + uDir * 1.3846153).rgb * 0.3162162;
      s += texture2D(tDiffuse, vUv - uDir * 1.3846153).rgb * 0.3162162;
      s += texture2D(tDiffuse, vUv + uDir * 3.2307692).rgb * 0.0702702;
      s += texture2D(tDiffuse, vUv - uDir * 3.2307692).rgb * 0.0702702;
      gl_FragColor = vec4(s, 1.0);
    }`,depthTest:!1,depthWrite:!1}),w=new S({uniforms:{tColor:{value:null},tND:{value:null},tB1:{value:null},tB2:{value:null},tB3:{value:null},tB4:{value:null},tB5:{value:null},uBW:{value:[.5,.35,.3,0,0]},uFogOn:{value:0},uFogColor:{value:new k(8425640)},uFogP:{value:new d(.02,.4,0,0)},uFogMax:{value:.9},uCamPos:{value:new h},uCamRot:{value:new q},uProj:{value:new d(1,1,0,0)},uRayN:{value:0},uRayPos:{value:[new d,new d,new d,new d]},uRayCol:{value:[new h,new h,new h,new h]},uRayP:{value:new d(.8,.4,.96,.9)},uRaySamples:{value:24},uTexel:{value:new R},uLineOn:{value:1},uLineW:{value:1},uDepthTh:{value:.012},uNormalTh:{value:.22},uLineOp:{value:.7},uLineTint:{value:new k(1841710)},uAO:{value:.45},uAORad:{value:.35},uFocalPx:{value:1e3},uBloom:{value:.55},uExposure:{value:1},uContrast:{value:1},uSat:{value:1},uGamma:{value:1},uLift:{value:new h},uGain:{value:new h(1,1,1)},uVig:{value:.25},uGuard:{value:0},uAsp:{value:1},uDiagLines:{value:0},tVol:{value:null},uVolOn:{value:0},uVolCol:{value:new h}},defines:{POST_LINES:1,POST_AO:1,POST_RAYS:1},vertexShader:L,fragmentShader:`
    varying vec2 vUv;
    uniform sampler2D tColor, tND, tB1, tB2, tB3, tB4, tB5;
    uniform float uBW[5];
    uniform float uFogOn, uFogMax, uRaySamples;
    uniform vec3 uFogColor, uCamPos;
    uniform vec4 uFogP, uProj, uRayP;
    uniform mat3 uCamRot;
    uniform int uRayN;
    uniform vec4 uRayPos[4];
    uniform vec3 uRayCol[4];
    uniform vec2 uTexel;
    uniform float uLineOn, uLineW, uDepthTh, uNormalTh, uLineOp, uAO, uAORad, uFocalPx, uBloom;
    uniform float uExposure, uContrast, uSat, uGamma, uVig, uGuard, uAsp, uDiagLines;
    uniform vec3 uLineTint, uLift, uGain;
    uniform sampler2D tVol;
    uniform float uVolOn;
    uniform vec3 uVolCol;

    vec4 nd(vec2 uv){ return texture2D(tND, uv); }
    void main(){
      vec3 col = texture2D(tColor, vUv).rgb;
      vec4 c0 = nd(vUv);
      float i0 = c0.a;                       // inverse depth, 0 = background
      float bg = step(i0, 1e-6);
      float edge = 0.0;
      #if POST_LINES
      if (uLineOn > 0.5 && bg < 0.5) {
        vec2 o = uTexel * uLineW;
        vec4 cL = nd(vUv + vec2(-o.x, 0.0)), cR = nd(vUv + vec2(o.x, 0.0));
        vec4 cU = nd(vUv + vec2(0.0, o.y)),  cD = nd(vUv + vec2(0.0, -o.y));
        vec4 c1 = nd(vUv + vec2(-o.x, -o.y)), c2 = nd(vUv + vec2(o.x, o.y));
        vec4 c3 = nd(vUv + vec2(-o.x, o.y)),  c4 = nd(vUv + vec2(o.x, -o.y));
        /* relative second difference of inverse depth: scale-free creases and steps */
        float dx = abs(cL.a + cR.a - 2.0 * i0), dy = abs(cU.a + cD.a - 2.0 * i0);
        float d1 = abs(c1.a + c2.a - 2.0 * i0), d2 = abs(c3.a + c4.a - 2.0 * i0);
        float dE = max(max(dx, dy), max(d1, d2) * 0.7) / i0;
        float depthEdge = smoothstep(uDepthTh, uDepthTh * 2.5, dE);
        /* silhouette: only the NEARER side of a depth step draws the line */
        float farthest = min(min(cL.a, cR.a), min(cU.a, cD.a));
        float sil = smoothstep(uDepthTh * 4.0, uDepthTh * 9.0, (i0 - farthest) / i0);
        vec3 n0 = c0.rgb * 2.0 - 1.0;
        float ne = 0.0;
        ne = max(ne, 1.0 - dot(n0, cL.rgb * 2.0 - 1.0));
        ne = max(ne, 1.0 - dot(n0, cR.rgb * 2.0 - 1.0));
        ne = max(ne, 1.0 - dot(n0, cU.rgb * 2.0 - 1.0));
        ne = max(ne, 1.0 - dot(n0, cD.rgb * 2.0 - 1.0));
        float normEdge = smoothstep(uNormalTh, uNormalTh * 2.2, ne);
        edge = clamp(max(max(depthEdge, sil), normEdge), 0.0, 1.0) * uLineOp;
        float lum0 = dot(col, vec3(0.299, 0.587, 0.114));
        edge *= 1.0 - smoothstep(0.7, 1.6, lum0) * 0.85;
      }
      #endif
      /* screen-space AO from the inverse-depth buffer */
      float ao = 1.0;
      #if POST_AO
      if (uAO > 0.0 && bg < 0.5) {
        float d0 = 1.0 / i0;
        float rpx = clamp(uAORad * uFocalPx / d0, 2.0, 60.0);
        float occ = 0.0;
        for (int k = 0; k < 10; k++) {
          float fk = float(k);
          float a = fk * 2.39996 + 0.6;
          float r = rpx * (0.25 + 0.75 * (fk + 0.5) / 10.0);
          vec2 uv = vUv + vec2(cos(a), sin(a)) * r * uTexel;
          float is = textureLod(tND, uv, 0.0).a;
          float diff = d0 - 1.0 / max(is, 1e-6);   // >0: sample is in front
          float w = step(1e-6, is) * smoothstep(0.02 * d0, 0.06 * d0 + 0.02, diff) * (1.0 - smoothstep(uAORad, uAORad * 3.0, diff));
          occ += w;
        }
        ao = 1.0 - uAO * occ / 10.0;
      }
      #endif
      col *= ao;
      vec3 line = col * 0.3 + uLineTint * 0.45;
      col = mix(col, line, edge);
      if (uDiagLines > 0.5) col = mix(vec3(0.93), uLineTint, edge / max(uLineOp, 1e-3));
      /* height fog: analytic integral of exp(-falloff*(y-base)) along the eye ray */
      if (uFogOn > 0.5 && bg < 0.5) {
        float dist = 1.0 / i0;
        vec2 ndc = vUv * 2.0 - 1.0;
        vec3 vray = vec3((ndc.x + uProj.z) / uProj.x, (ndc.y + uProj.w) / uProj.y, -1.0);
        vec3 wp = uCamPos + uCamRot * (vray * dist);
        float len = length(wp - uCamPos);
        float dy = wp.y - uCamPos.y;
        float k = max(uFogP.y, 1e-4);
        float e0 = exp(-k * (uCamPos.y - uFogP.z));
        float e1 = exp(-k * (wp.y - uFogP.z));
        float tau = abs(dy) < 1e-3 ? e0 * len : (e0 - e1) / (k * dy) * len;
        tau = max(tau, 0.0) * uFogP.x;
        tau *= smoothstep(0.0, 1.0, (len - uFogP.w) / max(len, 1e-3));
        col = mix(col, uFogColor, min(1.0 - exp(-tau), uFogMax));
      }
      /* volumetric light: the half-resolution march (VOL), blurred */
      if (uVolOn > 0.5) col += uVolCol * texture2D(tVol, vUv).r;
      /* light shafts: march from the pixel toward each projected light, collecting bright pixels */
      #if POST_RAYS
      for (int li = 0; li < 4; li++) {
        if (li >= uRayN) break;
        vec4 rp = uRayPos[li];
        vec2 dl = (rp.xy - vUv) * uRayP.w / uRaySamples;
        vec2 suv = vUv;
        float illum = 1.0, acc = 0.0;
        for (int s = 0; s < 48; s++) {
          if (float(s) >= uRaySamples) break;
          suv += dl;
          vec3 sc = textureLod(tColor, clamp(suv, 0.001, 0.999), 0.0).rgb;
          float m = smoothstep(uRayP.x, uRayP.x + uRayP.y, dot(sc, vec3(0.299, 0.587, 0.114)));
          acc += m * illum;
          illum *= uRayP.z;
        }
        col += uRayCol[li] * (acc / uRaySamples) * rp.z * rp.w;
      }
      #endif
      vec3 b = texture2D(tB1, vUv).rgb * uBW[0] + texture2D(tB2, vUv).rgb * uBW[1] + texture2D(tB3, vUv).rgb * uBW[2]
             + texture2D(tB4, vUv).rgb * uBW[3] + texture2D(tB5, vUv).rgb * uBW[4];
      col += b * uBloom;
      /* grade: exposure, lift/gain, gamma, contrast around mid grey, saturation */
      col *= uExposure;
      col = col * uGain + uLift * (1.0 - col);
      col = pow(max(col, 0.0), vec3(1.0 / uGamma));
      col = (col - 0.18) * uContrast + 0.18;
      float l = dot(col, vec3(0.2126, 0.7152, 0.0722));
      col = mix(vec3(l), col, uSat);
      col.g = mix(col.g, max(col.g, min(col.r, col.b)), uGuard);
      vec2 vv = vUv - 0.5; vv.x *= uAsp * 0.66;
      col *= 1.0 - dot(vv, vv) * uVig * 2.0;
      gl_FragColor = vec4(max(col, 0.0), 1.0);
    }`,depthTest:!1,depthWrite:!1}),K=new S({uniforms:{tND:{value:null},tShadow:{value:null},uShadowM:{value:new ve},uCamPos:{value:new h},uCamRot:{value:new q},uProj:{value:new d(1,1,0,0)},uLight:{value:new d(0,1,0,0)},uRange:{value:0},uP:{value:new d(.05,.6,.5,40)},uSteps:{value:24},uBias:{value:.002},uTop:{value:1e4},uBottom:{value:-1e4},uFloor:{value:0}},vertexShader:L,fragmentShader:`
    #include <packing>
    varying vec2 vUv;
    uniform sampler2D tND, tShadow;
    uniform mat4 uShadowM;
    uniform vec3 uCamPos;
    uniform mat3 uCamRot;
    uniform vec4 uProj, uLight, uP;
    uniform float uRange, uSteps, uBias, uTop, uBottom, uFloor;
    float lit(vec3 wp){
      vec4 s = uShadowM * vec4(wp, 1.0);
      s.xyz /= s.w;
      if (s.w <= 0.0 || s.x < 0.0 || s.x > 1.0 || s.y < 0.0 || s.y > 1.0 || s.z > 1.0) return 0.0;
      float l = step(s.z - uBias, unpackRGBAToDepth(texture2D(tShadow, s.xy)));
      if (uLight.w > 0.5) {
        l *= smoothstep(0.5, 0.36, length(s.xy - 0.5));      /* the spot's cone: its map's inscribed circle */
        if (uRange > 0.0) { float k = clamp(1.0 - distance(wp, uLight.xyz) / uRange, 0.0, 1.0); l *= k * k; }
      }
      return l;
    }
    void main(){
      float i0 = texture2D(tND, vUv).a;
      vec2 ndc = vUv * 2.0 - 1.0;
      vec3 vray = vec3((ndc.x + uProj.z) / uProj.x, (ndc.y + uProj.w) / uProj.y, -1.0);
      vec3 dir = normalize(uCamRot * vray);
      float len = i0 > 1e-6 ? length(vray) / i0 : uP.w;      /* eye-ray length to the surface (view depth x ray scale) */
      len = min(len, uP.w);
      float n = max(uSteps, 1.0);
      float jit = fract(52.9829189 * fract(0.06711056 * gl_FragCoord.x + 0.00583715 * gl_FragCoord.y));
      float acc = 0.0;
      for (int i = 0; i < 64; i++) {
        if (float(i) >= n) break;
        vec3 sp = uCamPos + dir * len * ((float(i) + jit) / n);
        if (sp.y < uTop && sp.y > uBottom) acc += lit(sp);
      }
      vec3 tl = uLight.w > 0.5 ? normalize(uLight.xyz - uCamPos) : uLight.xyz;
      float g = uP.z, c = dot(dir, tl);
      float hg = (1.0 - g * g) / pow(max(1.0 + g * g - 2.0 * g * c, 1e-4), 1.5); /* 1 when isotropic (g = 0) */
      float illum = acc / n * len * uP.x;
      float o = 1.0 - exp(-illum * hg);
      o = max(o - uFloor, 0.0) / max(1.0 - uFloor, 1e-3);
      gl_FragColor = vec4(min(o, uP.y), 0.0, 0.0, 1.0);
    }`,depthTest:!1,depthWrite:!1}),I=new S({uniforms:{tDiffuse:{value:null},uTexel:{value:new R},uStrength:{value:.6},uRadius:{value:2}},vertexShader:L,fragmentShader:`
    varying vec2 vUv; uniform sampler2D tDiffuse; uniform vec2 uTexel; uniform float uStrength, uRadius;
    void main(){
      vec3 c0 = textureLod(tDiffuse, vUv, 0.0).rgb;
      int R = int(uRadius);
      vec3 m[4]; vec3 s[4]; float n[4];
      for (int i = 0; i < 4; i++) { m[i] = vec3(0.0); s[i] = vec3(0.0); n[i] = 0.0; }
      for (int j = -4; j <= 4; j++) for (int i = -4; i <= 4; i++) {
        if (abs(i) > R || abs(j) > R) continue;
        vec3 c = textureLod(tDiffuse, vUv + vec2(float(i), float(j)) * uTexel, 0.0).rgb;
        c = min(c, vec3(8.0));
        if (i <= 0 && j <= 0) { m[0] += c; s[0] += c * c; n[0] += 1.0; }
        if (i >= 0 && j <= 0) { m[1] += c; s[1] += c * c; n[1] += 1.0; }
        if (i >= 0 && j >= 0) { m[2] += c; s[2] += c * c; n[2] += 1.0; }
        if (i <= 0 && j >= 0) { m[3] += c; s[3] += c * c; n[3] += 1.0; }
      }
      float best = 1e9; vec3 res = c0;
      for (int k = 0; k < 4; k++) {
        vec3 mu = m[k] / n[k];
        vec3 v = abs(s[k] / n[k] - mu * mu);
        float sig = v.r + v.g + v.b;
        if (sig < best) { best = sig; res = mu; }
      }
      gl_FragColor = vec4(mix(c0, res, uStrength), 1.0);
    }`,depthTest:!1,depthWrite:!1}),B=new S({uniforms:{tComp:{value:null},tND:{value:null},uTexel:{value:new R},uDof:{value:0},uFocus:{value:4},uRange:{value:2},uNearK:{value:1},uFarK:{value:1},uMaxR:{value:6},uGrain:{value:.012},uTime:{value:0},uTiltC:{value:new R(.5,.5)},uTiltR:{value:new R(.5,.5)},uTiltSoft:{value:.25},uTiltS:{value:0},uWarp:{value:new d},uWarpC:{value:new R(.5,.5)}},defines:{POST_DOF:1},vertexShader:L,fragmentShader:`
    varying vec2 vUv;
    uniform sampler2D tComp, tND;
    uniform vec2 uTexel, uTiltC, uTiltR;
    uniform float uDof, uFocus, uRange, uNearK, uFarK, uMaxR, uGrain, uTime, uTiltSoft, uTiltS;
    uniform vec4 uWarp;
    uniform vec2 uWarpC;
    /* warp: what lies beyond uWarp.w metres (the world outside a car, not its cabin) */
    float far(vec2 uv){ float i = textureLod(tND, uv, 0.0).a; return i < 1e-6 ? 1.0 : smoothstep(uWarp.w, uWarp.w * 2.0, 1.0 / i); }
    float whash(vec2 p){ vec3 q = fract(vec3(p.xyx) * 0.1031); q += dot(q, q.yzx + 33.33); return fract((q.x + q.y) * q.z); }
    float tilt(vec2 uv){
      vec2 q = (uv - uTiltC) / uTiltR;
      return smoothstep(1.0, 1.0 + uTiltSoft / min(uTiltR.x, uTiltR.y), length(q)) * uTiltS;
    }
    float coc(float inv){
      if (inv < 1e-6) return uMaxR * uFarK;         // sky/background: far
      float d = 1.0 / inv;
      float x = d - uFocus;
      float k = x < 0.0 ? uNearK : uFarK;
      return clamp((abs(x) - uRange * 0.5) / max(uRange, 1e-3), 0.0, 1.0) * uMaxR * k;
    }
    void main(){
      vec3 col = textureLod(tComp, vUv, 0.0).rgb;
      float c0 = max(coc(textureLod(tND, vUv, 0.0).a), tilt(vUv)) * step(0.5, uDof);
      vec3 acc = col; float wsum = 1.0;
      #if POST_DOF
      for (int k = 1; k < 28; k++) {
        float fk = float(k);
        float r = sqrt(fk / 28.0) * c0;
        float a = fk * 2.39996;
        vec2 uv = clamp(vUv + vec2(cos(a), sin(a)) * r * uTexel, vec2(0.0), vec2(1.0));
        float cs = max(coc(textureLod(tND, uv, 0.0).a), tilt(uv));
        float w = step(0.5, c0) * smoothstep(r - 1.0, r + 0.5, max(cs, c0 * 0.6));
        acc += textureLod(tComp, uv, 0.0).rgb * w; wsum += w;
      }
      #endif
      col = acc / wsum;
      /* speed (after warp-shooter's post.js, MIT): radial blur toward the focus of expansion with a little chromatic
         split, speed lines and an exposure flash. The blur and lines take only far pixels (and far samples), so a
         cabin stays sharp while the street streams past it. */
      if (uWarp.x + uWarp.y + uWarp.z > 0.001) {
        vec2 d = vUv - uWarpC;
        float r = length(d * vec2(uTexel.y / uTexel.x, 1.0));
        float m0 = far(vUv);
        if (uWarp.x > 0.001 && m0 > 0.0) {
          vec3 wa = vec3(0.0); float ww = 0.0;
          for (int i = 0; i < 12; i++) {
            float t = float(i) / 11.0;
            float k = 1.0 - uWarp.x * t * 0.12, ab = uWarp.x * t * 0.006;
            vec2 uv = uWarpC + d * k;
            float w = (1.0 - t * 0.6) * far(uv);
            wa += vec3(textureLod(tComp, uWarpC + d * k * (1.0 + ab), 0.0).r, textureLod(tComp, uv, 0.0).g, textureLod(tComp, uWarpC + d * k * (1.0 - ab), 0.0).b) * w;
            ww += w;
          }
          if (ww > 0.0) col = mix(col, wa / ww, m0 * smoothstep(0.04, 0.35, r));
        }
        if (uWarp.y > 0.001) {
          float seg = atan(d.y, d.x * uTexel.y / uTexel.x) * 38.197;
          float id = floor(seg), h = whash(vec2(id, floor(uTime * 20.0))), h2 = whash(vec2(id * 1.37, 7.0));
          float line = step(0.72, h) * (1.0 - smoothstep(0.0, 0.16 + 0.1 * h2, abs(fract(seg) - 0.5))) * smoothstep(0.32 + 0.3 * h2, 0.95, r);
          col += vec3(0.8, 0.92, 1.0) * line * uWarp.y * 0.55 * m0;
        }
        /* the flash over-exposes the street (the eyes adjusting to the bridge): it blows out to white, the cabin is untouched */
        float fl = uWarp.z * m0; /* the street outside only: the cabin is never flashed */
        col = col * (1.0 + 3.0 * fl) + vec3(0.55, 0.53, 0.5) * fl;
      }
      float g = fract(sin(dot(vUv * 1000.0, vec2(12.9898, 78.233))) * 43758.5453);
      col += (g - 0.5) * uGrain;
      gl_FragColor = vec4(col, 1.0);
      #include <tonemapping_fragment>
      #include <colorspace_fragment>
    }`,depthTest:!1,depthWrite:!1});class xe{constructor(t){this.r=t,this.params=J({}),this.fsCam=new te(-1,1,1,-1,0,1),this.fsScene=new ae,this.fsMesh=new le(new ue(2,2),w),this.fsMesh.frustumCulled=!1,this.fsScene.add(this.fsMesh),this.w=this.h=2,this.rtColor=b(2,2,{samples:4}),this.rtND=b(2,2,{samples:0}),this.rtComp=b(2,2,{depth:!1}),this.rtPaint=b(2,2,{depth:!1}),this.rtVol=b(2,2,{depth:!1}),this.rtVol2=b(2,2,{depth:!1}),this.bl=Array.from({length:5},()=>b(2,2,{depth:!1})),this.bt=Array.from({length:5},()=>b(2,2,{depth:!1})),this._m3=new q,this._v=new h,this.diagLines=!1,this.warp={amount:0,lines:0,flash:0,near:2.5,center:[.5,.5]},this._l0=new oe}setParams(t){this.params=J(t)}setFeatures(t={}){const a=(u,e,r)=>{const s=r===!1?0:1;u.defines[e]!==s&&(u.defines[e]=s,u.needsUpdate=!0)};a(w,"POST_LINES",t.outline),a(w,"POST_AO",t.ao),a(w,"POST_RAYS",t.rays),a(B,"POST_DOF",t.dof)}materials(){return{comp:w,final:B,bright:_,blur:f,paint:I,vol:K}}gbufferMaterial(t){return X(t)}setSize(t,a){this.w=t,this.h=a,this.rtColor.setSize(t,a),this.rtND.setSize(t,a),this.rtComp.setSize(t,a),this.rtPaint.setSize(t,a),this.rtVol.setSize(Math.max(1,t>>1),Math.max(1,a>>1)),this.rtVol2.setSize(Math.max(1,t>>1),Math.max(1,a>>1));for(let u=0;u<5;u++){const e=2<<u;this.bl[u].setSize(Math.max(1,Math.floor(t/e)),Math.max(1,Math.floor(a/e))),this.bt[u].setSize(Math.max(1,Math.floor(t/e)),Math.max(1,Math.floor(a/e)))}}blit(t,a){this.fsMesh.material=t,this.r.setRenderTarget(a||null),this.r.render(this.fsScene,this.fsCam)}render(t,a,u=0){const e=this.r,r=this.params;if(!r.enabled){e.setRenderTarget(null),e.render(t,a);return}e.shadowMap.needsUpdate=!0,e.setRenderTarget(this.rtColor),e.render(t,a);const s=t.background,D=t.fog,j=e.getClearColor(new k),z=e.getClearAlpha();t.background=null,t.fog=null;const m=this._swap||(this._swap=[]);m.length=0;const p=this._hide||(this._hide=[]);p.length=0,t.traverseVisible(o=>{if(o.layers.test(this._l0)){if(o.isPoints||o.isLine||o.isSprite){p.push(o);return}if(o.isMesh){if(Array.isArray(o.material)||o.material.transparent&&!o.material.depthWrite||o.material.blending===ie){p.push(o);return}m.push(o,o.material),o.material=X(o.material)}}});for(const o of p)o.visible=!1;const A=a.layers.mask;a.layers.disable(fe),e.setClearColor(8421631,0),e.setRenderTarget(this.rtND),e.clear();const c=e.shadowMap.enabled;e.shadowMap.enabled=!1,e.render(t,a),e.shadowMap.enabled=c,a.layers.mask=A;for(let o=0;o<m.length;o+=2)m[o].material=m[o+1];m.length=0;for(const o of p)o.visible=!0;p.length=0,t.background=s,t.fog=D,e.setClearColor(j,z);const x=r.bloom;if(x.enabled&&x.strength>0){_.uniforms.tDiffuse.value=this.rtColor.texture,_.uniforms.uTh.value=x.threshold,_.uniforms.uKnee.value=x.knee,this.blit(_,this.bl[0]);const o=Math.max(3,Math.min(5,(x.weights||[]).reduce((v,y,U)=>y>0?U+1:v,3)));for(let v=0;v<o;v++){const y=v===0?this.bl[0]:this.bl[v-1];v>0&&(f.uniforms.tDiffuse.value=y.texture,f.uniforms.uDir.value.set(0,0),this.blit(f,this.bl[v]));const U=this.bl[v].width,G=this.bl[v].height;for(let N=0;N<2;N++)f.uniforms.tDiffuse.value=this.bl[v].texture,f.uniforms.uDir.value.set(1/U,0),this.blit(f,this.bt[v]),f.uniforms.tDiffuse.value=this.bt[v].texture,f.uniforms.uDir.value.set(0,1/G),this.blit(f,this.bl[v])}}const l=w.uniforms,M=r.outline,V=r.ao,g=r.grade;l.tColor.value=this.rtColor.texture,l.tND.value=this.rtND.texture,l.tB1.value=this.bl[0].texture,l.tB2.value=this.bl[1].texture,l.tB3.value=this.bl[2].texture,l.tB4.value=this.bl[3].texture,l.tB5.value=this.bl[4].texture;const ee=x.weights||[.5,.35,.3,0,0];l.uBW.value=[0,1,2,3,4].map(o=>ee[o]||0),a.updateMatrixWorld();const T=r.fog;l.uFogOn.value=T.enabled&&a.isPerspectiveCamera?1:0,l.uFogColor.value.set(T.color),l.uFogP.value.set(T.density,T.falloff,T.base,T.near),l.uFogMax.value=T.maxOpacity,l.uCamPos.value.setFromMatrixPosition(a.matrixWorld),l.uCamRot.value.setFromMatrix4(a.matrixWorld);const W=a.projectionMatrix.elements;l.uProj.value.set(W[0],W[5],W[8],W[9]);const P=r.rays,$=P.enabled&&a.isPerspectiveCamera?P.lights.slice(0,4):[];l.uRayN.value=$.length,$.forEach((o,v)=>{const y=this._v.set(...o.pos).project(a),U=this._v.set(...o.pos).applyMatrix4(a.matrixWorldInverse).z>0,G=Math.max(Math.abs(y.x),Math.abs(y.y)),N=U?0:1-Math.min(1,Math.max(0,(G-1)/.6));l.uRayPos.value[v].set(y.x*.5+.5,y.y*.5+.5,o.strength??.3,N);const E=new k(o.color??16769200);l.uRayCol.value[v].set(E.r,E.g,E.b)}),l.uVolOn.value=this.volume(t,a,r.vol)?1:0,l.uRayP.value.set(P.threshold,P.knee,P.decay,P.density),l.uRaySamples.value=Math.min(48,P.samples),l.uTexel.value.set(1/this.w,1/this.h),l.uLineOn.value=M.enabled?1:0,l.uLineW.value=M.width*Math.max(1,this.h/1080),l.uDepthTh.value=M.depthTh,l.uNormalTh.value=M.normalTh,l.uLineOp.value=M.opacity,l.uLineTint.value.set(M.color),l.uAO.value=V.enabled?V.strength:0,l.uAORad.value=V.radius,l.uFocalPx.value=a.isPerspectiveCamera?this.h*.5/Math.tan(a.fov*Math.PI/360):1e3,l.uBloom.value=x.enabled?x.strength:0,l.uExposure.value=g.exposure,l.uContrast.value=g.contrast,l.uSat.value=g.saturation,l.uGamma.value=g.gamma,l.uLift.value.set(...g.lift),l.uGain.value.set(...g.gain),l.uVig.value=g.vignette,l.uGuard.value=g.guard,l.uAsp.value=this.w/this.h,l.uDiagLines.value=this.diagLines?1:0,this.blit(w,this.rtComp);let H=this.rtComp;if(r.paint.enabled&&r.paint.strength>0){const o=I.uniforms;o.tDiffuse.value=this.rtComp.texture,o.uTexel.value.set(1/this.w,1/this.h),o.uStrength.value=r.paint.strength,o.uRadius.value=Math.min(4,Math.max(1,Math.round(r.paint.radius*Math.max(1,this.h/1080)))),this.blit(I,this.rtPaint),H=this.rtPaint}const n=B.uniforms,C=r.dof;n.tComp.value=H.texture,n.tND.value=this.rtND.texture,n.uTexel.value.set(1/this.w,1/this.h),n.uDof.value=C.enabled?1:0,n.uFocus.value=C.focus,n.uRange.value=C.range,n.uNearK.value=C.near,n.uFarK.value=C.far,n.uMaxR.value=C.maxRadius*Math.max(1,this.h/1080);const F=C.tilt||{};n.uTiltC.value.set(...F.center||[.5,.5]),n.uTiltR.value.set(...F.radius||[.5,.5]),n.uTiltSoft.value=F.soft!==void 0?F.soft:.25,n.uTiltS.value=(F.strength||0)*Math.max(1,this.h/1080),n.uGrain.value=g.staticGrain||0,n.uTime.value=u;const O=this.warp;n.uWarp.value.set(O.amount,O.lines,O.flash,O.near),n.uWarpC.value.set(...O.center),this.blit(B,null)}volume(t,a,u){var p,A;if(!u.enabled||!a.isPerspectiveCamera)return!1;let e=this._volLight;(!e||e.parent===null||!e.visible||!e.castShadow)&&(e=null,t.traverseVisible(c=>{!e&&c.castShadow&&(c.isDirectionalLight||c.isSpotLight)&&c.intensity>0&&(e=c)}),this._volLight=e);const r=(A=(p=e==null?void 0:e.shadow)==null?void 0:p.map)==null?void 0:A.texture;if(!r||!this.r.shadowMap.enabled)return!1;const s=K.uniforms,D=w.uniforms;if(s.tND.value=this.rtND.texture,s.tShadow.value=r,s.uShadowM.value.copy(e.shadow.matrix),s.uCamPos.value.copy(D.uCamPos.value),s.uCamRot.value.copy(D.uCamRot.value),s.uProj.value.copy(D.uProj.value),e.isSpotLight)s.uLight.value.set(...e.getWorldPosition(this._v).toArray(),1);else{e.target.updateMatrixWorld();const c=e.getWorldPosition(this._v).sub(e.target.getWorldPosition(new h)).normalize();s.uLight.value.set(c.x,c.y,c.z,0)}s.uRange.value=e.isSpotLight?e.distance||e.shadow.camera.far:0,s.uP.value.set(u.density,u.max,u.phase,u.dist),s.uSteps.value=Math.min(64,u.steps),s.uBias.value=u.bias,s.uTop.value=u.top??1e4,s.uBottom.value=u.bottom??-1e4,s.uFloor.value=Math.min(.95,u.floor||0),this.blit(K,this.rtVol);const j=this.rtVol.width,z=this.rtVol.height;f.uniforms.tDiffuse.value=this.rtVol.texture,f.uniforms.uDir.value.set(1/j,0),this.blit(f,this.rtVol2),f.uniforms.tDiffuse.value=this.rtVol2.texture,f.uniforms.uDir.value.set(0,1/z),this.blit(f,this.rtVol),D.tVol.value=this.rtVol.texture;const m=u.color!==null&&u.color!==void 0?new k(u.color):e.color;return D.uVolCol.value.set(m.r,m.g,m.b).multiplyScalar(u.strength),!0}dispose(){[this.rtColor,this.rtND,this.rtComp,this.rtPaint,this.rtVol,this.rtVol2,...this.bl,...this.bt].forEach(t=>t.dispose()),this.fsMesh.geometry.dispose()}}export{ce as DEFAULT_POST,xe as Post,J as mergePost};
