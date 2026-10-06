import{V as c,t as G,J as h,d as b,C as k,A as O,b as C,M,r as R,P as F}from"./index-DNaBlw4_.js";import{L as A,J as L,S as q}from"./texlocal-dECVdLjB.js";import"./react-three-fiber.esm-cABpMmTN.js";function H(e={}){return{sunDir:[.1,-.7,.4],zenith:1189482,horizon:5798072,ground:1846360,glow:9087712,haze:.85,gradPow:.6,stars:3,moon:2.4,moonDir:[.1,.5,-.85],clouds:{cover:.28,alt:.9,scale:.6,soft:.6,dark:.8,high:.5,lit:11846888,shade:2767984},...e}}const K="varying vec3 vWorld; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vWorld = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }",W=`
  varying vec3 vWorld;
  uniform float uGain, uSil, uSilK;
  uniform vec3 uSilCol;
  ${q}
  /* far tree line and roof ridge as a function of azimuth: pine spikes over a low swell */
  float silHeight(float az){
    float h = 0.045 + 0.04 * skyNoise(vec2(az * 5.0, 1.7));
    float zig = abs(fract(az * 11.0 + skyNoise(vec2(az * 3.0, 2.0)) * 0.8) - 0.5) * 2.0;
    h += (1.0 - zig) * 0.05 * smoothstep(0.35, 0.7, skyNoise(vec2(az * 4.0, 8.0)));
    float roof = smoothstep(0.55, 0.6, skyNoise(vec2(az * 2.2, 5.0)));       /* a tiled roof line, straight-edged */
    h = mix(h, 0.075 + 0.02 * abs(fract(az * 7.0) - 0.5), roof * 0.6);
    return h;
  }
  void main(){
    vec3 d = normalize(vWorld - cameraPosition);
    vec3 c = skyColor(d);
    if (uSil > 0.0) {
      float az = atan(d.x, -d.z);
      float hs = silHeight(az) * uSilK;
      float below = 1.0 - smoothstep(hs - 0.004, hs + 0.004, d.y);
      float mist = smoothstep(hs + 0.05, hs, d.y) * 0.35;                  /* ground mist lifts the foot of the trees */
      c = mix(c, uSilCol, below * uSil);
      c += vec3(0.1, 0.14, 0.2) * mist * (1.0 - below) * uSil;
    }
    /* a denser, steady star field than the core sky draws (a window only shows a few degrees of it) */
    vec3 q = d * 520.0; vec3 cq = floor(q); float rs = skyHash(cq);
    float st = step(0.982, rs) * smoothstep(0.55, 0.0, length(fract(q) - 0.5)) * (0.45 + 0.55 * skyHash(cq + 7.0));
    c += vec3(0.85, 0.92, 1.0) * st * 1.2 * smoothstep(0.0, 0.12, d.y) * (1.0 - uSil * smoothstep(0.0, 0.1, 0.2 - d.y) * 0.0);
    gl_FragColor = vec4(c * uGain, 1.0);
  }`;function Y(e={}){const i=L(H(e.sky)),f=new b({vertexShader:K,fragmentShader:W,fog:!1,side:C,uniforms:{...i,uGain:{value:e.gain??1.4},uSil:{value:e.sil??0},uSilK:{value:e.silK??1},uSilCol:{value:new k(e.silColor??198158)}}}),o=e.round?new R(e.w/2,40):new F(e.w,e.h),t=new M(o,f);return t.name=e.name||"moon-window",e.pos&&t.position.set(...e.pos),e.rot&&t.rotation.set(...e.rot),t.layers.set(A),t.userData.dynamic=!0,t.userData.update=l=>{i.uTime.value=l},t.userData.dispose=()=>{o.dispose(),f.dispose()},t}const E=`
  varying vec2 vUv; varying vec3 vN; varying vec3 vV;
  void main(){
    vUv = uv;
    vec4 w = modelMatrix * vec4(position, 1.0);
    vN = normalize(mat3(modelMatrix) * normal);
    vV = cameraPosition - w.xyz;
    gl_Position = projectionMatrix * viewMatrix * w;
  }`,B=`
  uniform vec3 uColor; uniform float uOpacity, uTime, uSeed;
  varying vec2 vUv; varying vec3 vN; varying vec3 vV;
  float h1(float x){ return fract(sin(x * 127.1 + uSeed) * 43758.5453); }
  float n1(float x){ float i = floor(x), f = fract(x); f = f * f * (3.0 - 2.0 * f); return mix(h1(i), h1(i + 1.0), f); }
  void main(){
    float across = abs(vUv.x - 0.5) * 2.0;
    float soft = pow(max(1.0 - across * across, 0.0), 2.0);          /* soft across the strip: no hard edge anywhere */
    float face = pow(clamp(abs(dot(normalize(vN), normalize(vV))), 0.0001, 1.0), 1.3);     /* a strip seen edge-on fades out instead of becoming a line */
    float streak = 0.6 + 0.4 * n1(vUv.x * 9.0 + uTime * 0.01) * (0.6 + 0.4 * n1(vUv.x * 27.0 + 3.0));
    float dust = 0.88 + 0.12 * n1(vUv.y * 9.0 - uTime * 0.03 + vUv.x * 5.0);
    float fall = pow(max(1.0 - vUv.y, 0.0), 1.1) * smoothstep(0.0, 0.05, vUv.y);   /* v 0 at the source, 1 at the far end */
    gl_FragColor = vec4(uColor, uOpacity * soft * face * streak * dust * fall);
  }`;function $(e={}){const i=new c(...e.from),o=new c(...e.to).clone().sub(i),t=o.length();o.normalize();const l=Math.abs(o.y)>.9?new c(1,0,0):new c(0,1,0).cross(o).normalize(),T=o.clone().cross(l).normalize(),U=e.r0??.5,N=e.r1??1,u=[],p=[],y=[],g=[],w=3;for(let n=0;n<w;n++){const x=n/w*Math.PI,S=l.clone().multiplyScalar(Math.cos(x)).addScaledVector(T,Math.sin(x)),v=S.clone().cross(o).normalize(),r=u.length/3;for(const[P,V,_]of[[U,0,0],[N,1,1]]){const D=i.clone().addScaledVector(o,t*V);for(const z of[0,1]){const m=D.clone().addScaledVector(S,(z-.5)*2*P);u.push(m.x,m.y,m.z),p.push(v.x,v.y,v.z),y.push(z,_)}}g.push(r,r+1,r+2,r+1,r+3,r+2)}const s=new G;s.setAttribute("position",new h(u,3)),s.setAttribute("normal",new h(p,3)),s.setAttribute("uv",new h(y,2)),s.setIndex(g);const d=new b({vertexShader:E,fragmentShader:B,transparent:!0,depthWrite:!1,side:C,fog:!1,blending:O,uniforms:{uColor:{value:new k(e.color??9087200)},uOpacity:{value:e.opacity??.12},uTime:{value:0},uSeed:{value:e.seed??3}}}),a=new M(s,d);return a.name=e.name||"moon-shaft",a.renderOrder=6,a.frustumCulled=!1,a.layers.set(A),a.userData.dynamic=!0,a.userData.update=n=>{d.uniforms.uTime.value=n},a.userData.dispose=()=>{s.dispose(),d.dispose()},a}export{$ as lightShaft,H as nightSky,Y as skyPortal};
