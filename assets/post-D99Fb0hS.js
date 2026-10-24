import{az as Z,aA as ee,e as te,P as ae,x as w,aB as I,F as d,aC as ue,z as _,y as le,V as P,I as x,W as ie,aD as j,aE as se,H as oe}from"./index-BTUI7STM.js";import{L as re}from"./texlocal-CEzmuU6F.js";import"./react-three-fiber.esm-Ca6MHwNG.js";const F=`
varying vec2 vUv;
void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,ne={enabled:!0,outline:{enabled:!0,color:1841710,opacity:.7,width:1,depthTh:.012,normalTh:.22},ao:{enabled:!0,strength:.45,radius:.35},bloom:{enabled:!0,threshold:.85,knee:.35,strength:.55,weights:[.5,.35,.3,0,0]},fog:{enabled:!1,color:8425640,density:.02,falloff:.4,base:0,near:0,maxOpacity:.9},rays:{enabled:!1,lights:[],threshold:.8,knee:.4,decay:.96,density:.9,samples:24},paint:{enabled:!1,strength:.6,radius:2},grade:{exposure:1,contrast:1,saturation:1,lift:[0,0,0],gain:[1,1,1],gamma:1,vignette:.25,grain:0,staticGrain:0,guard:0},dof:{enabled:!1,focus:4,range:2,near:1,far:1,maxRadius:6,tilt:{center:[.5,.5],radius:[.5,.5],soft:.25,strength:0}}};function $(i,e){const t=Array.isArray(i)?i.slice():{...i};if(!e)return t;for(const s of Object.keys(e)){const l=e[s];l&&typeof l=="object"&&!Array.isArray(l)&&typeof i[s]=="object"&&!Array.isArray(i[s])?t[s]=$(i[s],l):t[s]=l}return t}function z(i){return $(ne,i||{})}function R(i,e,t={}){const s=new ie(i,e,{type:oe,format:se,minFilter:j,magFilter:j,depthBuffer:t.depth!==!1,stencilBuffer:!1,generateMipmaps:!1});return t.samples&&(s.samples=t.samples),s}const ve=`
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
  }`,fe=`
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
  }`,K=new Map;function V(i){const e=i.alphaTest>0&&(i.map||i.alphaMap),t=e?i.map||i.alphaMap:null,s=`${i.side}|${e?t.id+":"+i.alphaTest:""}|${i.polygonOffset}|${i.polygonOffsetFactor}|${i.polygonOffsetUnits}|${i.depthTest}|${i.depthWrite}`;let l=K.get(s);return l||(l=new w({vertexShader:ve,fragmentShader:fe,side:i.side,polygonOffset:i.polygonOffset,polygonOffsetFactor:i.polygonOffsetFactor,polygonOffsetUnits:i.polygonOffsetUnits,depthTest:i.depthTest,depthWrite:i.depthWrite,defines:e?{ND_ALPHA:"",USE_MAP:"",MAP_UV:"uv"}:{},uniforms:e?{uMap:{value:t},uAlphaTest:{value:i.alphaTest},mapTransform:{value:t.matrix}}:{}}),K.set(s,l)),l}const M=new w({uniforms:{tDiffuse:{value:null},uTh:{value:.85},uKnee:{value:.35}},vertexShader:F,fragmentShader:`
    varying vec2 vUv; uniform sampler2D tDiffuse; uniform float uTh, uKnee;
    void main(){
      vec3 c = texture2D(tDiffuse, vUv).rgb;
      float l = max(max(c.r, c.g), c.b);
      float k = smoothstep(uTh, uTh + uKnee, l);
      gl_FragColor = vec4(min(c * k, vec3(16.0)), 1.0);
    }`,depthTest:!1,depthWrite:!1}),v=new w({uniforms:{tDiffuse:{value:null},uDir:{value:new P(1,0)}},vertexShader:F,fragmentShader:`
    varying vec2 vUv; uniform sampler2D tDiffuse; uniform vec2 uDir;
    void main(){
      vec3 s = texture2D(tDiffuse, vUv).rgb * 0.2270270;
      s += texture2D(tDiffuse, vUv + uDir * 1.3846153).rgb * 0.3162162;
      s += texture2D(tDiffuse, vUv - uDir * 1.3846153).rgb * 0.3162162;
      s += texture2D(tDiffuse, vUv + uDir * 3.2307692).rgb * 0.0702702;
      s += texture2D(tDiffuse, vUv - uDir * 3.2307692).rgb * 0.0702702;
      gl_FragColor = vec4(s, 1.0);
    }`,depthTest:!1,depthWrite:!1}),b=new w({uniforms:{tColor:{value:null},tND:{value:null},tB1:{value:null},tB2:{value:null},tB3:{value:null},tB4:{value:null},tB5:{value:null},uBW:{value:[.5,.35,.3,0,0]},uFogOn:{value:0},uFogColor:{value:new _(8425640)},uFogP:{value:new x(.02,.4,0,0)},uFogMax:{value:.9},uCamPos:{value:new d},uCamRot:{value:new I},uProj:{value:new x(1,1,0,0)},uRayN:{value:0},uRayPos:{value:[new x,new x,new x,new x]},uRayCol:{value:[new d,new d,new d,new d]},uRayP:{value:new x(.8,.4,.96,.9)},uRaySamples:{value:24},uTexel:{value:new P},uLineOn:{value:1},uLineW:{value:1},uDepthTh:{value:.012},uNormalTh:{value:.22},uLineOp:{value:.7},uLineTint:{value:new _(1841710)},uAO:{value:.45},uAORad:{value:.35},uFocalPx:{value:1e3},uBloom:{value:.55},uExposure:{value:1},uContrast:{value:1},uSat:{value:1},uGamma:{value:1},uLift:{value:new d},uGain:{value:new d(1,1,1)},uVig:{value:.25},uGuard:{value:0},uAsp:{value:1},uDiagLines:{value:0}},defines:{POST_LINES:1,POST_AO:1,POST_RAYS:1},vertexShader:F,fragmentShader:`
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
    }`,depthTest:!1,depthWrite:!1}),W=new w({uniforms:{tDiffuse:{value:null},uTexel:{value:new P},uStrength:{value:.6},uRadius:{value:2}},vertexShader:F,fragmentShader:`
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
    }`,depthTest:!1,depthWrite:!1}),L=new w({uniforms:{tComp:{value:null},tND:{value:null},uTexel:{value:new P},uDof:{value:0},uFocus:{value:4},uRange:{value:2},uNearK:{value:1},uFarK:{value:1},uMaxR:{value:6},uGrain:{value:.012},uTime:{value:0},uTiltC:{value:new P(.5,.5)},uTiltR:{value:new P(.5,.5)},uTiltSoft:{value:.25},uTiltS:{value:0}},defines:{POST_DOF:1},vertexShader:F,fragmentShader:`
    varying vec2 vUv;
    uniform sampler2D tComp, tND;
    uniform vec2 uTexel, uTiltC, uTiltR;
    uniform float uDof, uFocus, uRange, uNearK, uFarK, uMaxR, uGrain, uTime, uTiltSoft, uTiltS;
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
      float g = fract(sin(dot(vUv * 1000.0, vec2(12.9898, 78.233))) * 43758.5453);
      col += (g - 0.5) * uGrain;
      gl_FragColor = vec4(col, 1.0);
      #include <tonemapping_fragment>
      #include <colorspace_fragment>
    }`,depthTest:!1,depthWrite:!1});class me{constructor(e){this.r=e,this.params=z({}),this.fsCam=new Z(-1,1,1,-1,0,1),this.fsScene=new ee,this.fsMesh=new te(new ae(2,2),b),this.fsMesh.frustumCulled=!1,this.fsScene.add(this.fsMesh),this.w=this.h=2,this.rtColor=R(2,2,{samples:4}),this.rtND=R(2,2,{samples:0}),this.rtComp=R(2,2,{depth:!1}),this.rtPaint=R(2,2,{depth:!1}),this.bl=Array.from({length:5},()=>R(2,2,{depth:!1})),this.bt=Array.from({length:5},()=>R(2,2,{depth:!1})),this._m3=new I,this._v=new d,this.diagLines=!1,this._l0=new ue}setParams(e){this.params=z(e)}setFeatures(e={}){const t=(s,l,n)=>{const O=n===!1?0:1;s.defines[l]!==O&&(s.defines[l]=O,s.needsUpdate=!0)};t(b,"POST_LINES",e.outline),t(b,"POST_AO",e.ao),t(b,"POST_RAYS",e.rays),t(L,"POST_DOF",e.dof)}materials(){return{comp:b,final:L,bright:M,blur:v,paint:W}}gbufferMaterial(e){return V(e)}setSize(e,t){this.w=e,this.h=t,this.rtColor.setSize(e,t),this.rtND.setSize(e,t),this.rtComp.setSize(e,t),this.rtPaint.setSize(e,t);for(let s=0;s<5;s++){const l=2<<s;this.bl[s].setSize(Math.max(1,Math.floor(e/l)),Math.max(1,Math.floor(t/l))),this.bt[s].setSize(Math.max(1,Math.floor(e/l)),Math.max(1,Math.floor(t/l)))}}blit(e,t){this.fsMesh.material=e,this.r.setRenderTarget(t||null),this.r.render(this.fsScene,this.fsCam)}render(e,t,s=0){const l=this.r,n=this.params;if(!n.enabled){l.setRenderTarget(null),l.render(e,t);return}l.shadowMap.needsUpdate=!0,l.setRenderTarget(this.rtColor),l.render(e,t);const O=e.background,H=e.fog,Y=l.getClearColor(new _),q=l.getClearAlpha();e.background=null,e.fog=null;const y=this._swap||(this._swap=[]);y.length=0;const T=this._hide||(this._hide=[]);T.length=0,e.traverseVisible(u=>{if(u.layers.test(this._l0)){if(u.isPoints||u.isLine||u.isSprite){T.push(u);return}if(u.isMesh){if(Array.isArray(u.material)||u.material.transparent&&!u.material.depthWrite||u.material.blending===le){T.push(u);return}y.push(u,u.material),u.material=V(u.material)}}});for(const u of T)u.visible=!1;const J=t.layers.mask;t.layers.disable(re),l.setClearColor(8421631,0),l.setRenderTarget(this.rtND),l.clear();const Q=l.shadowMap.enabled;l.shadowMap.enabled=!1,l.render(e,t),l.shadowMap.enabled=Q,t.layers.mask=J;for(let u=0;u<y.length;u+=2)y[u].material=y[u+1];y.length=0;for(const u of T)u.visible=!0;T.length=0,e.background=O,e.fog=H,l.setClearColor(Y,q);const c=n.bloom;if(c.enabled&&c.strength>0){M.uniforms.tDiffuse.value=this.rtColor.texture,M.uniforms.uTh.value=c.threshold,M.uniforms.uKnee.value=c.knee,this.blit(M,this.bl[0]);const u=Math.max(3,Math.min(5,(c.weights||[]).reduce((o,h,S)=>h>0?S+1:o,3)));for(let o=0;o<u;o++){const h=o===0?this.bl[0]:this.bl[o-1];o>0&&(v.uniforms.tDiffuse.value=h.texture,v.uniforms.uDir.value.set(0,0),this.blit(v,this.bl[o]));const S=this.bl[o].width,k=this.bl[o].height;for(let A=0;A<2;A++)v.uniforms.tDiffuse.value=this.bl[o].texture,v.uniforms.uDir.value.set(1/S,0),this.blit(v,this.bt[o]),v.uniforms.tDiffuse.value=this.bt[o].texture,v.uniforms.uDir.value.set(0,1/k),this.blit(v,this.bl[o])}}const a=b.uniforms,D=n.outline,N=n.ao,f=n.grade;a.tColor.value=this.rtColor.texture,a.tND.value=this.rtND.texture,a.tB1.value=this.bl[0].texture,a.tB2.value=this.bl[1].texture,a.tB3.value=this.bl[2].texture,a.tB4.value=this.bl[3].texture,a.tB5.value=this.bl[4].texture;const X=c.weights||[.5,.35,.3,0,0];a.uBW.value=[0,1,2,3,4].map(u=>X[u]||0),t.updateMatrixWorld();const m=n.fog;a.uFogOn.value=m.enabled&&t.isPerspectiveCamera?1:0,a.uFogColor.value.set(m.color),a.uFogP.value.set(m.density,m.falloff,m.base,m.near),a.uFogMax.value=m.maxOpacity,a.uCamPos.value.setFromMatrixPosition(t.matrixWorld),a.uCamRot.value.setFromMatrix4(t.matrixWorld);const U=t.projectionMatrix.elements;a.uProj.value.set(U[0],U[5],U[8],U[9]);const p=n.rays,G=p.enabled&&t.isPerspectiveCamera?p.lights.slice(0,4):[];a.uRayN.value=G.length,G.forEach((u,o)=>{const h=this._v.set(...u.pos).project(t),S=this._v.set(...u.pos).applyMatrix4(t.matrixWorldInverse).z>0,k=Math.max(Math.abs(h.x),Math.abs(h.y)),A=S?0:1-Math.min(1,Math.max(0,(k-1)/.6));a.uRayPos.value[o].set(h.x*.5+.5,h.y*.5+.5,u.strength??.3,A);const B=new _(u.color??16769200);a.uRayCol.value[o].set(B.r,B.g,B.b)}),a.uRayP.value.set(p.threshold,p.knee,p.decay,p.density),a.uRaySamples.value=Math.min(48,p.samples),a.uTexel.value.set(1/this.w,1/this.h),a.uLineOn.value=D.enabled?1:0,a.uLineW.value=D.width*Math.max(1,this.h/1080),a.uDepthTh.value=D.depthTh,a.uNormalTh.value=D.normalTh,a.uLineOp.value=D.opacity,a.uLineTint.value.set(D.color),a.uAO.value=N.enabled?N.strength:0,a.uAORad.value=N.radius,a.uFocalPx.value=t.isPerspectiveCamera?this.h*.5/Math.tan(t.fov*Math.PI/360):1e3,a.uBloom.value=c.enabled?c.strength:0,a.uExposure.value=f.exposure,a.uContrast.value=f.contrast,a.uSat.value=f.saturation,a.uGamma.value=f.gamma,a.uLift.value.set(...f.lift),a.uGain.value.set(...f.gain),a.uVig.value=f.vignette,a.uGuard.value=f.guard,a.uAsp.value=this.w/this.h,a.uDiagLines.value=this.diagLines?1:0,this.blit(b,this.rtComp);let E=this.rtComp;if(n.paint.enabled&&n.paint.strength>0){const u=W.uniforms;u.tDiffuse.value=this.rtComp.texture,u.uTexel.value.set(1/this.w,1/this.h),u.uStrength.value=n.paint.strength,u.uRadius.value=Math.min(4,Math.max(1,Math.round(n.paint.radius*Math.max(1,this.h/1080)))),this.blit(W,this.rtPaint),E=this.rtPaint}const r=L.uniforms,g=n.dof;r.tComp.value=E.texture,r.tND.value=this.rtND.texture,r.uTexel.value.set(1/this.w,1/this.h),r.uDof.value=g.enabled?1:0,r.uFocus.value=g.focus,r.uRange.value=g.range,r.uNearK.value=g.near,r.uFarK.value=g.far,r.uMaxR.value=g.maxRadius*Math.max(1,this.h/1080);const C=g.tilt||{};r.uTiltC.value.set(...C.center||[.5,.5]),r.uTiltR.value.set(...C.radius||[.5,.5]),r.uTiltSoft.value=C.soft!==void 0?C.soft:.25,r.uTiltS.value=(C.strength||0)*Math.max(1,this.h/1080),r.uGrain.value=f.staticGrain||0,r.uTime.value=s,this.blit(L,null)}dispose(){[this.rtColor,this.rtND,this.rtComp,this.rtPaint,...this.bl,...this.bt].forEach(e=>e.dispose()),this.fsMesh.geometry.dispose()}}export{ne as DEFAULT_POST,me as Post,z as mergePost};
