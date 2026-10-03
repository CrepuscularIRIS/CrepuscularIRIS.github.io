import{C as d,P as T,I as M,o as b,f as C,g as P,V as G,M as O,G as U,b as W}from"./index-Dy76UFLB.js";import{L as z}from"./texlocal-eE-3lTCg.js";const A=`
  float h21(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
  float vn(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y); }
  float fbm(vec2 p){ float a = 0.5, s = 0.0; for (int i = 0; i < 4; i++) { s += a * vn(p); p = p * 2.03 + 11.7; a *= 0.5; } return s; }
`;function D(a){let s=a>>>0||1;return()=>(s=s*1664525+1013904223>>>0)/4294967296}function F(a={}){const{pos:s=[0,0,0],height:m=6,spread:v=1.4,size:g=1.2,count:n=28,speed:y=.12,wind:c=[.6,.2],seed:h=1,glow:x=0}=a,t=new d(a.color??9079446),f=new d(a.colorTop??a.color??13158612),p=a.opacity??.5,l=D(h),e=new T(1,1),o=new M;o.index=e.index,o.attributes.position=e.attributes.position,o.attributes.uv=e.attributes.uv;const r=new Float32Array(n),S=new Float32Array(n*2);for(let u=0;u<n;u++)r[u]=(u+l()*.8)/n,S[u*2]=l(),S[u*2+1]=l();o.setAttribute("aPhase",new b(r,1)),o.setAttribute("aSeed",new b(S,2)),o.instanceCount=n;const w=new C({transparent:!0,depthWrite:!1,fog:!1,uniforms:{uTime:{value:0},uPos:{value:new G(...s)},uHeight:{value:m},uSpread:{value:v},uSize:{value:g},uSpeed:{value:y},uWind:{value:new P(...c)},uColor:{value:t},uTop:{value:f},uOpacity:{value:p},uGlow:{value:x},uGlowColor:{value:new d(a.glowColor??16751184)}},vertexShader:`
      attribute float aPhase; attribute vec2 aSeed;
      uniform float uTime, uHeight, uSpread, uSize, uSpeed; uniform vec3 uPos; uniform vec2 uWind;
      varying vec2 vUv; varying float vAge; varying vec2 vSeed;
      void main(){
        float age = fract(uTime * uSpeed + aPhase);
        vAge = age; vSeed = aSeed; vUv = uv;
        vec3 c = uPos;
        c.y += age * uHeight;
        c.xz += uWind * age * age * uHeight * 0.5;
        float ang = aSeed.x * 6.2831 + uTime * 0.15 * (aSeed.y - 0.5);
        c.xz += vec2(cos(ang), sin(ang)) * uSpread * age * (0.3 + aSeed.y);
        float sz = uSize * (0.35 + age * 2.4) * (0.7 + 0.6 * aSeed.y);
        vec4 mv = viewMatrix * vec4(c, 1.0);
        mv.xy += position.xy * sz;
        gl_Position = projectionMatrix * mv;
      }`,fragmentShader:`
      uniform vec3 uColor, uTop, uGlowColor; uniform float uOpacity, uGlow, uTime;
      varying vec2 vUv; varying float vAge; varying vec2 vSeed;
      ${A}
      void main(){
        vec2 p = vUv - 0.5;
        float r = length(p) * 2.0;
        float n = fbm(p * 3.2 + vSeed * 17.0 + vec2(0.0, -uTime * 0.05));
        float blob = smoothstep(1.0, 0.15, r + (n - 0.5) * 0.9);
        float life = smoothstep(0.0, 0.08, vAge) * pow(1.0 - vAge, 1.4);
        float a = blob * life * uOpacity;
        vec3 col = mix(uColor, uTop, smoothstep(0.0, 0.8, vAge));
        col = mix(col, uGlowColor, uGlow * (1.0 - smoothstep(0.0, 0.3, vAge)) * 0.8);
        col *= 0.75 + 0.35 * n;
        gl_FragColor = vec4(col, a);
      }`}),i=new O(o,w);return i.name="smokePlume",i.frustumCulled=!1,i.renderOrder=8,i.layers.set(z),i.userData.dynamic=!0,i.userData.update=u=>{w.uniforms.uTime.value=u},i.userData.dispose=()=>{o.dispose(),w.dispose()},i}function H(a={}){var p,l;const{size:s=[60,60],y:m=.3,layers:v=4,height:g=1.4,scale:n=.12,speed:y=.01,seed:c=2}=a,h=new d(a.color??13160672),x=a.opacity??.22,t=new U;t.name="groundMist";const f=[];for(let e=0;e<v;e++){const o=new C({transparent:!0,depthWrite:!1,fog:!1,side:W,uniforms:{uTime:{value:0},uColor:{value:h},uOpacity:{value:x/Math.sqrt(v)},uScale:{value:n},uSpeed:{value:y*(.7+.6*((e*7+c)%5)/5)},uOff:{value:e*13.7+c}},vertexShader:"varying vec2 vUv; varying vec3 vW; void main(){ vUv = uv; vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }",fragmentShader:`
        uniform vec3 uColor; uniform float uTime, uOpacity, uScale, uSpeed, uOff; varying vec2 vUv; varying vec3 vW;
        ${A}
        void main(){
          vec2 e = min(vUv, 1.0 - vUv); float edge = smoothstep(0.0, 0.25, min(e.x, e.y));
          float n = fbm(vW.xz * uScale + vec2(uTime * uSpeed, uOff)) * 0.7 + fbm(vW.xz * uScale * 2.7 - vec2(uOff, uTime * uSpeed * 1.6)) * 0.3;
          float d = smoothstep(0.38, 0.8, n);
          gl_FragColor = vec4(uColor, d * edge * uOpacity);
        }`});f.push(o);const r=new O(new T(s[0],s[1]),o);r.rotation.x=-Math.PI/2,r.position.y=m+(v>1?e/(v-1)*g:0),r.renderOrder=7,r.layers.set(z),t.add(r)}return t.position.set(((p=a.pos)==null?void 0:p[0])??0,0,((l=a.pos)==null?void 0:l[1])??0),t.userData.dynamic=!0,t.userData.update=e=>{for(const o of f)o.uniforms.uTime.value=e},t.userData.dispose=()=>{t.traverse(e=>{e.isMesh&&(e.geometry.dispose(),e.material.dispose())})},t}export{H as g,F as s};
