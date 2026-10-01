import{O as W,S as V,ar as z,av as H,o as b,L as j,q as N,p as q,d as A,e as x,w as I,J as U,h as $,H as J}from"./index-DDlL8vX0.js";import{W as Y}from"./WorldCanvas-C3CKJGsq.js";const T=`
varying vec2 vUv;
void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,Q={enabled:!0,outline:{enabled:!0,color:1841710,opacity:.7,width:1,depthTh:.012,normalTh:.22},ao:{enabled:!0,strength:.45,radius:.35},bloom:{enabled:!0,threshold:.85,knee:.35,strength:.55},grade:{exposure:1,contrast:1,saturation:1,lift:[0,0,0],gain:[1,1,1],gamma:1,vignette:.25,grain:0},dof:{enabled:!1,focus:4,range:2,near:1,far:1,maxRadius:6,tilt:{center:[.5,.5],radius:[.5,.5],soft:.25,strength:0}}};function S(r,t){const l=Array.isArray(r)?r.slice():{...r};if(!t)return l;for(const i of Object.keys(t)){const a=t[i];a&&typeof a=="object"&&!Array.isArray(a)&&typeof r[i]=="object"&&!Array.isArray(r[i])?l[i]=S(r[i],a):l[i]=a}return l}function w(r){return S(Q,r||{})}function o(r,t,l={}){const i=new I(r,t,{type:J,format:$,minFilter:U,magFilter:U,depthBuffer:l.depth!==!1,stencilBuffer:!1,generateMipmaps:!1});return l.samples&&(i.samples=l.samples),i}const X=`
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
  }`,Z=`
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
  }`,M=new Map;function ee(r){const t=r.alphaTest>0&&(r.map||r.alphaMap),l=t?r.map||r.alphaMap:null,i=`${r.side}|${t?l.id+":"+r.alphaTest:""}`;let a=M.get(i);return a||(a=new b({vertexShader:X,fragmentShader:Z,side:r.side,defines:t?{ND_ALPHA:"",USE_MAP:"",MAP_UV:"uv"}:{},uniforms:t?{uMap:{value:l},uAlphaTest:{value:r.alphaTest},mapTransform:{value:l.matrix}}:{}}),M.set(i,a)),a}const D=new b({uniforms:{tDiffuse:{value:null},uTh:{value:.85},uKnee:{value:.35}},vertexShader:T,fragmentShader:`
    varying vec2 vUv; uniform sampler2D tDiffuse; uniform float uTh, uKnee;
    void main(){
      vec3 c = texture2D(tDiffuse, vUv).rgb;
      float l = max(max(c.r, c.g), c.b);
      float k = smoothstep(uTh, uTh + uKnee, l);
      gl_FragColor = vec4(min(c * k, vec3(16.0)), 1.0);
    }`,depthTest:!1,depthWrite:!1}),n=new b({uniforms:{tDiffuse:{value:null},uDir:{value:new x(1,0)}},vertexShader:T,fragmentShader:`
    varying vec2 vUv; uniform sampler2D tDiffuse; uniform vec2 uDir;
    void main(){
      vec3 s = texture2D(tDiffuse, vUv).rgb * 0.2270270;
      s += texture2D(tDiffuse, vUv + uDir * 1.3846153).rgb * 0.3162162;
      s += texture2D(tDiffuse, vUv - uDir * 1.3846153).rgb * 0.3162162;
      s += texture2D(tDiffuse, vUv + uDir * 3.2307692).rgb * 0.0702702;
      s += texture2D(tDiffuse, vUv - uDir * 3.2307692).rgb * 0.0702702;
      gl_FragColor = vec4(s, 1.0);
    }`,depthTest:!1,depthWrite:!1}),C=new b({uniforms:{tColor:{value:null},tND:{value:null},tB1:{value:null},tB2:{value:null},tB3:{value:null},uTexel:{value:new x},uLineOn:{value:1},uLineW:{value:1},uDepthTh:{value:.012},uNormalTh:{value:.22},uLineOp:{value:.7},uLineTint:{value:new N(1841710)},uAO:{value:.45},uAORad:{value:.35},uFocalPx:{value:1e3},uBloom:{value:.55},uExposure:{value:1},uContrast:{value:1},uSat:{value:1},uGamma:{value:1},uLift:{value:new A},uGain:{value:new A(1,1,1)},uVig:{value:.25},uAsp:{value:1},uDiagLines:{value:0}},vertexShader:T,fragmentShader:`
    varying vec2 vUv;
    uniform sampler2D tColor, tND, tB1, tB2, tB3;
    uniform vec2 uTexel;
    uniform float uLineOn, uLineW, uDepthTh, uNormalTh, uLineOp, uAO, uAORad, uFocalPx, uBloom;
    uniform float uExposure, uContrast, uSat, uGamma, uVig, uAsp, uDiagLines;
    uniform vec3 uLineTint, uLift, uGain;

    vec4 nd(vec2 uv){ return texture2D(tND, uv); }
    void main(){
      vec3 col = texture2D(tColor, vUv).rgb;
      vec4 c0 = nd(vUv);
      float i0 = c0.a;                       // inverse depth, 0 = background
      float bg = step(i0, 1e-6);
      float edge = 0.0;
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
      /* screen-space AO from the inverse-depth buffer */
      float ao = 1.0;
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
      col *= ao;
      vec3 line = col * 0.3 + uLineTint * 0.45;
      col = mix(col, line, edge);
      if (uDiagLines > 0.5) col = mix(vec3(0.93), uLineTint, edge / max(uLineOp, 1e-3));
      vec3 b = texture2D(tB1, vUv).rgb * 0.5 + texture2D(tB2, vUv).rgb * 0.35 + texture2D(tB3, vUv).rgb * 0.3;
      col += b * uBloom;
      /* grade: exposure, lift/gain, gamma, contrast around mid grey, saturation */
      col *= uExposure;
      col = col * uGain + uLift * (1.0 - col);
      col = pow(max(col, 0.0), vec3(1.0 / uGamma));
      col = (col - 0.18) * uContrast + 0.18;
      float l = dot(col, vec3(0.2126, 0.7152, 0.0722));
      col = mix(vec3(l), col, uSat);
      vec2 vv = vUv - 0.5; vv.x *= uAsp * 0.66;
      col *= 1.0 - dot(vv, vv) * uVig * 2.0;
      gl_FragColor = vec4(max(col, 0.0), 1.0);
    }`,depthTest:!1,depthWrite:!1}),R=new b({uniforms:{tComp:{value:null},tND:{value:null},uTexel:{value:new x},uDof:{value:0},uFocus:{value:4},uRange:{value:2},uNearK:{value:1},uFarK:{value:1},uMaxR:{value:6},uGrain:{value:.012},uTime:{value:0},uTiltC:{value:new x(.5,.5)},uTiltR:{value:new x(.5,.5)},uTiltSoft:{value:.25},uTiltS:{value:0}},vertexShader:T,fragmentShader:`
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
      for (int k = 1; k < 28; k++) {
        float fk = float(k);
        float r = sqrt(fk / 28.0) * c0;
        float a = fk * 2.39996;
        vec2 uv = clamp(vUv + vec2(cos(a), sin(a)) * r * uTexel, vec2(0.0), vec2(1.0));
        float cs = max(coc(textureLod(tND, uv, 0.0).a), tilt(uv));
        float w = step(0.5, c0) * smoothstep(r - 1.0, r + 0.5, max(cs, c0 * 0.6));
        acc += textureLod(tComp, uv, 0.0).rgb * w; wsum += w;
      }
      col = acc / wsum;
      float g = fract(sin(dot(vUv * 1000.0 + fract(uTime * 0.37), vec2(12.9898, 78.233))) * 43758.5453);
      col += (g - 0.5) * uGrain;
      gl_FragColor = vec4(col, 1.0);
      #include <colorspace_fragment>
    }`,depthTest:!1,depthWrite:!1});class le{constructor(t){this.r=t,this.params=w({}),this.fsCam=new W(-1,1,1,-1,0,1),this.fsScene=new V,this.fsMesh=new z(new H(2,2),C),this.fsMesh.frustumCulled=!1,this.fsScene.add(this.fsMesh),this.w=this.h=2,this.rtColor=o(2,2,{samples:4}),this.rtND=o(2,2,{samples:0}),this.rtComp=o(2,2,{depth:!1}),this.bl=[o(2,2,{depth:!1}),o(2,2,{depth:!1}),o(2,2,{depth:!1})],this.bt=[o(2,2,{depth:!1}),o(2,2,{depth:!1}),o(2,2,{depth:!1})],this.diagLines=!1,this._l0=new j}setParams(t){this.params=w(t)}setSize(t,l){this.w=t,this.h=l,this.rtColor.setSize(t,l),this.rtND.setSize(t,l),this.rtComp.setSize(t,l);for(let i=0;i<3;i++){const a=2<<i;this.bl[i].setSize(Math.max(1,Math.floor(t/a)),Math.max(1,Math.floor(l/a))),this.bt[i].setSize(Math.max(1,Math.floor(t/a)),Math.max(1,Math.floor(l/a)))}}blit(t,l){this.fsMesh.material=t,this.r.setRenderTarget(l||null),this.r.render(this.fsScene,this.fsCam)}render(t,l,i=0){const a=this.r,c=this.params;if(!c.enabled){a.setRenderTarget(null),a.render(t,l);return}a.shadowMap.needsUpdate=!0,a.setRenderTarget(this.rtColor),a.render(t,l);const _=t.background,k=t.fog,F=a.getClearColor(new N),O=a.getClearAlpha();t.background=null,t.fog=null;const h=this._swap||(this._swap=[]);h.length=0;const d=this._hide||(this._hide=[]);d.length=0,t.traverseVisible(e=>{if(e.layers.test(this._l0)){if(e.isPoints||e.isLine||e.isSprite){d.push(e);return}if(e.isMesh){if(Array.isArray(e.material)||e.material.transparent&&!e.material.depthWrite||e.material.blending===q){d.push(e);return}h.push(e,e.material),e.material=ee(e.material)}}});for(const e of d)e.visible=!1;const B=l.layers.mask;l.layers.disable(Y),a.setClearColor(8421631,0),a.setRenderTarget(this.rtND),a.clear();const P=a.shadowMap.enabled;a.shadowMap.enabled=!1,a.render(t,l),a.shadowMap.enabled=P,l.layers.mask=B;for(let e=0;e<h.length;e+=2)h[e].material=h[e+1];h.length=0;for(const e of d)e.visible=!0;d.length=0,t.background=_,t.fog=k,a.setClearColor(F,O);const m=c.bloom;if(m.enabled&&m.strength>0){D.uniforms.tDiffuse.value=this.rtColor.texture,D.uniforms.uTh.value=m.threshold,D.uniforms.uKnee.value=m.knee,this.blit(D,this.bl[0]);for(let e=0;e<3;e++){const E=e===0?this.bl[0]:this.bl[e-1];e>0&&(n.uniforms.tDiffuse.value=E.texture,n.uniforms.uDir.value.set(0,0),this.blit(n,this.bl[e]));const G=this.bl[e].width,K=this.bl[e].height;for(let y=0;y<2;y++)n.uniforms.tDiffuse.value=this.bl[e].texture,n.uniforms.uDir.value.set(1/G,0),this.blit(n,this.bt[e]),n.uniforms.tDiffuse.value=this.bt[e].texture,n.uniforms.uDir.value.set(0,1/K),this.blit(n,this.bl[e])}}const u=C.uniforms,p=c.outline,L=c.ao,v=c.grade;u.tColor.value=this.rtColor.texture,u.tND.value=this.rtND.texture,u.tB1.value=this.bl[0].texture,u.tB2.value=this.bl[1].texture,u.tB3.value=this.bl[2].texture,u.uTexel.value.set(1/this.w,1/this.h),u.uLineOn.value=p.enabled?1:0,u.uLineW.value=p.width*Math.max(1,this.h/1080),u.uDepthTh.value=p.depthTh,u.uNormalTh.value=p.normalTh,u.uLineOp.value=p.opacity,u.uLineTint.value.set(p.color),u.uAO.value=L.enabled?L.strength:0,u.uAORad.value=L.radius,u.uFocalPx.value=l.isPerspectiveCamera?this.h*.5/Math.tan(l.fov*Math.PI/360):1e3,u.uBloom.value=m.enabled?m.strength:0,u.uExposure.value=v.exposure,u.uContrast.value=v.contrast,u.uSat.value=v.saturation,u.uGamma.value=v.gamma,u.uLift.value.set(...v.lift),u.uGain.value.set(...v.gain),u.uVig.value=v.vignette,u.uAsp.value=this.w/this.h,u.uDiagLines.value=this.diagLines?1:0,this.blit(C,this.rtComp);const s=R.uniforms,f=c.dof;s.tComp.value=this.rtComp.texture,s.tND.value=this.rtND.texture,s.uTexel.value.set(1/this.w,1/this.h),s.uDof.value=f.enabled?1:0,s.uFocus.value=f.focus,s.uRange.value=f.range,s.uNearK.value=f.near,s.uFarK.value=f.far,s.uMaxR.value=f.maxRadius*Math.max(1,this.h/1080);const g=f.tilt||{};s.uTiltC.value.set(...g.center||[.5,.5]),s.uTiltR.value.set(...g.radius||[.5,.5]),s.uTiltSoft.value=g.soft!==void 0?g.soft:.25,s.uTiltS.value=(g.strength||0)*Math.max(1,this.h/1080),s.uGrain.value=0,s.uTime.value=i,this.blit(R,null)}dispose(){[this.rtColor,this.rtND,this.rtComp,...this.bl,...this.bt].forEach(t=>t.dispose()),this.fsMesh.geometry.dispose()}}export{Q as DEFAULT_POST,le as Post,w as mergePost};
