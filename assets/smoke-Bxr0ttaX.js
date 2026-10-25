import{z as m,P as O,$ as G,a0 as C,x as D,V as U,F as W,e as A,G as N,X as E}from"./index-QIr3aeYT.js";import{L as P}from"./texlocal-C3GyaHRn.js";const M=`
  float h21(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
  float vn(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y); }
  float fbm(vec2 p){ float a = 0.5, s = 0.0; for (int i = 0; i < 4; i++) { s += a * vn(p); p = p * 2.03 + 11.7; a *= 0.5; } return s; }
`;function _(e){let s=e>>>0||1;return()=>(s=s*1664525+1013904223>>>0)/4294967296}function H(e={}){const{pos:s=[0,0,0],height:g=6,spread:v=1.4,size:y=1.2,count:n=28,speed:h=.12,wind:c=[.6,.2],seed:x=1,glow:S=0}=e,t=new m(e.color??9079446),f=new m(e.colorTop??e.color??13158612),p=e.opacity??.5,d=e.decay??1.4,a=e.edge??.9,l=e.near??0,o=_(x),w=new O(1,1),r=new G;r.index=w.index,r.attributes.position=w.attributes.position,r.attributes.uv=w.attributes.uv;const z=new Float32Array(n),b=new Float32Array(n*2);for(let i=0;i<n;i++)z[i]=(i+o()*.8)/n,b[i*2]=o(),b[i*2+1]=o();r.setAttribute("aPhase",new C(z,1)),r.setAttribute("aSeed",new C(b,2)),r.instanceCount=n;const T=new D({transparent:!0,depthWrite:!1,fog:!1,uniforms:{uTime:{value:0},uPos:{value:new W(...s)},uHeight:{value:g},uSpread:{value:v},uSize:{value:y},uSpeed:{value:h},uWind:{value:new U(...c)},uColor:{value:t},uTop:{value:f},uOpacity:{value:p},uGlow:{value:S},uGlowColor:{value:new m(e.glowColor??16751184)},uDecay:{value:d},uEdge:{value:a},uNear:{value:l}},vertexShader:`
      attribute float aPhase; attribute vec2 aSeed;
      uniform float uTime, uHeight, uSpread, uSize, uSpeed; uniform vec3 uPos; uniform vec2 uWind;
      varying vec2 vUv; varying float vAge; varying vec2 vSeed; varying float vDepth;
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
        vDepth = -mv.z;
        gl_Position = projectionMatrix * mv;
      }`,fragmentShader:`
      uniform vec3 uColor, uTop, uGlowColor; uniform float uOpacity, uGlow, uTime, uDecay, uEdge, uNear;
      varying vec2 vUv; varying float vAge; varying vec2 vSeed; varying float vDepth;
      ${M}
      void main(){
        vec2 p = vUv - 0.5;
        float r = length(p) * 2.0;
        float n = fbm(p * 3.2 + vSeed * 17.0 + vec2(0.0, -uTime * 0.05));
        float blob = smoothstep(1.0, 0.15, r + (n - 0.5) * uEdge);
        float life = smoothstep(0.0, 0.08, vAge) * pow(1.0 - vAge, uDecay);
        float a = blob * life * uOpacity;
        if (uNear > 0.0) a *= smoothstep(uNear * 0.35, uNear, vDepth);
        vec3 col = mix(uColor, uTop, smoothstep(0.0, 0.8, vAge));
        col = mix(col, uGlowColor, uGlow * (1.0 - smoothstep(0.0, 0.3, vAge)) * 0.8);
        col *= 0.75 + 0.35 * n;
        gl_FragColor = vec4(col, a);
      }`}),u=new A(r,T);return u.name="smokePlume",u.frustumCulled=!1,u.renderOrder=8,u.layers.set(P),u.userData.dynamic=!0,u.userData.update=i=>{T.uniforms.uTime.value=i},u.userData.dispose=()=>{r.dispose(),T.dispose()},u}function L(e={}){var p,d;const{size:s=[60,60],y:g=.3,layers:v=4,height:y=1.4,scale:n=.12,speed:h=.01,seed:c=2}=e,x=new m(e.color??13160672),S=e.opacity??.22,t=new N;t.name="groundMist";const f=[];for(let a=0;a<v;a++){const l=new D({transparent:!0,depthWrite:!1,fog:!1,side:E,uniforms:{uTime:{value:0},uColor:{value:x},uOpacity:{value:S/Math.sqrt(v)},uScale:{value:n},uSpeed:{value:h*(.7+.6*((a*7+c)%5)/5)},uOff:{value:a*13.7+c}},vertexShader:"varying vec2 vUv; varying vec3 vW; void main(){ vUv = uv; vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }",fragmentShader:`
        uniform vec3 uColor; uniform float uTime, uOpacity, uScale, uSpeed, uOff; varying vec2 vUv; varying vec3 vW;
        ${M}
        void main(){
          vec2 e = min(vUv, 1.0 - vUv); float edge = smoothstep(0.0, 0.25, min(e.x, e.y));
          float n = fbm(vW.xz * uScale + vec2(uTime * uSpeed, uOff)) * 0.7 + fbm(vW.xz * uScale * 2.7 - vec2(uOff, uTime * uSpeed * 1.6)) * 0.3;
          float d = smoothstep(0.38, 0.8, n);
          gl_FragColor = vec4(uColor, d * edge * uOpacity);
        }`});f.push(l);const o=new A(new O(s[0],s[1]),l);o.rotation.x=-Math.PI/2,o.position.y=g+(v>1?a/(v-1)*y:0),o.renderOrder=7,o.layers.set(P),t.add(o)}return t.position.set(((p=e.pos)==null?void 0:p[0])??0,0,((d=e.pos)==null?void 0:d[1])??0),t.userData.dynamic=!0,t.userData.update=a=>{for(const l of f)l.uniforms.uTime.value=a},t.userData.dispose=()=>{t.traverse(a=>{a.isMesh&&(a.geometry.dispose(),a.material.dispose())})},t}export{L as g,H as s};
