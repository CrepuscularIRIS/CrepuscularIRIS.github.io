import{t as V,u as S,d as g,V as _,w as O,C as T,A as W,M as w,P as z,k as X,b as C}from"./index-hs-7eMLX.js";import{g as y,r as M,n as h}from"./texlocal-CJ51Bynp.js";import{s as B}from"./smoke-DIyU98sP.js";import{m as b}from"./motes-zQKcKc8n.js";import{W as v}from"./uw-Db2q372T.js";import"./react-three-fiber.esm-96qyf323.js";const P=`
  float h21(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
  float vn(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y); }
`,A=`
  float caustic(vec2 p0, float t){
    vec2 pp = mod(p0, 6.2831853) - 250.0, ii = pp; float cc = 1.0;
    for (int n = 0; n < 4; n++) {
      float t2 = (t + 20.0) * (1.0 - 3.5 / float(n + 1));
      ii = pp + vec2(cos(t2 - ii.x) + sin(t2 + ii.y), sin(t2 - ii.y) + cos(t2 + ii.x));
      cc += 1.0 / length(vec2(pp.x / (sin(ii.x + t2) / 0.006), pp.y / (cos(ii.y + t2) / 0.006)));
    }
    cc /= 4.0; cc = 1.17 - pow(cc, 1.4);
    return pow(abs(cc), 8.0);
  }
`,F=(s,n)=>new g({transparent:!0,depthWrite:!1,blending:W,side:C,fog:!1,uniforms:{uT:{value:0},uColor:{value:new T(s)},uI:{value:n},uSeed:{value:0},uTop:{value:v.surf},uBot:{value:v.floor}},vertexShader:"varying vec3 vW; varying vec3 vN; varying vec2 vUv; void main(){ vUv = uv; vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; vN = normalize(mat3(modelMatrix) * normal); gl_Position = projectionMatrix * viewMatrix * w; }",fragmentShader:`
    uniform float uT, uI, uSeed, uTop, uBot; uniform vec3 uColor; varying vec3 vW; varying vec3 vN; varying vec2 vUv;
    ${P}
    void main(){
      vec3 V = normalize(cameraPosition - vW);
      float facing = abs(dot(normalize(vN), V));
      float edge = pow(facing, 1.6);                              /* the cone fades toward its silhouette: reads as a volume */
      float v = clamp((vW.y - uBot) / (uTop - uBot), 0.0, 1.0);    /* 1 at the surface */
      float vert = pow(v, 1.5) * smoothstep(0.0, 0.08, 1.0 - v);   /* strongest under the surface, gone at the floor */
      float band = 0.55 + 0.45 * vn(vec2(vUv.x * 6.0 + uSeed, v * 5.0 - uT * 0.05));      /* drifting light and dark bands along the shaft */
      float pulse = 0.7 + 0.3 * sin(uT * 0.18 + uSeed * 4.0);
      float near = smoothstep(0.4, 2.5, distance(cameraPosition, vW));   /* no hard slab when the camera nears a shaft */
      gl_FragColor = vec4(uColor * edge * vert * band * pulse * uI * near, 1.0);
    }`});function k(s,n,{count:t=16,seed:a=7}={}){const r=y(s,"godRays"),e=M(a),i=v.surf-v.floor+.6,l=[];for(let o=0;o<t;o++){const u=.18+e()*.42,f=u*(1.8+e()*1.2),c=F(o%3===0?12580095:9427199,.2+e()*.16);c.uniforms.uSeed.value=e()*20,l.push(c);const d=new w(new X(u,f,i,24,1,!0),c),m=-7+o/(t-1)*14+(e()-.5)*1.4,p=-1.5-e()*11,x=.14+e()*.06;d.position.set(m-Math.sin(x)*i*.5,v.surf-i*.5*Math.cos(x),p+i*.12),d.rotation.set(-.22,0,x),d.renderOrder=4,d.frustumCulled=!1,r.add(h(d))}return n.push(o=>{for(const u of l)u.uniforms.uT.value=o}),r}function D(s,n,t,{floorY:a=v.floor+.12}={}){const r=y(s,"bubbleColumns"),e=M(31);for(const[i,l,o,u]of t){const f=new V,c=new Float32Array(o*4);for(let p=0;p<c.length;p++)c[p]=e();f.setAttribute("position",new S(new Float32Array(o*3),3)),f.setAttribute("aSeed",new S(c,4));const d=new g({transparent:!0,depthWrite:!1,fog:!1,uniforms:{uT:{value:0},uV:{value:new _(i,a,l)},uTop:{value:v.surf-.05},uSize:{value:u},uScale:{value:540}},vertexShader:`
        attribute vec4 aSeed; uniform float uT, uTop, uSize, uScale; uniform vec3 uV; varying float vA; varying float vH;
        void main(){
          float rise = 0.02 + 0.018 * aSeed.w;                         /* slow, bigger bubbles faster */
          float u = fract(aSeed.x + uT * rise * (0.7 + aSeed.y));
          float h = u * (uTop - uV.y);
          vec3 p = uV + vec3(0.0, h, 0.0);
          float w = 0.04 + 0.22 * u;                                  /* column widens as it rises */
          p.x += (aSeed.y - 0.5) * w * 2.0 + sin(uT * 0.9 + aSeed.z * 30.0 + u * 8.0) * 0.05 * (0.3 + u);
          p.z += (aSeed.z - 0.5) * w * 2.0 + cos(uT * 0.8 + aSeed.w * 30.0 + u * 7.0) * 0.05 * (0.3 + u);
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * mv;
          float rad = uSize * (0.45 + 0.9 * aSeed.w) * (0.6 + 0.8 * u);
          gl_PointSize = clamp(rad * uScale / max(0.3, -mv.z), 1.5, 26.0);
          vA = smoothstep(0.0, 0.04, u) * (1.0 - smoothstep(0.93, 1.0, u));
          vH = u;
        }`,fragmentShader:`
        varying float vA; varying float vH;
        void main(){
          vec2 p = gl_PointCoord - 0.5; float d = length(p) * 2.0;
          float rim = smoothstep(0.55, 0.9, d) * (1.0 - smoothstep(0.92, 1.0, d));     /* thin bright shell */
          float spec = smoothstep(0.35, 0.0, length(p - vec2(-0.16, 0.18)));          /* window glint */
          float body = (1.0 - smoothstep(0.85, 1.0, d)) * 0.08;
          float a = (rim * 0.75 + spec * 0.8 + body) * vA;
          gl_FragColor = vec4(mix(vec3(0.5, 0.82, 1.0), vec3(0.92, 0.99, 1.0), spec + rim * 0.4) , a);
        }`}),m=new O(f,d);m.frustumCulled=!1,m.renderOrder=6,r.add(h(m)),n.push(p=>{d.uniforms.uT.value=p})}return r}function G(s,n){const t=y(s,"silt");for(const[e,i,l,o,u]of[[-3,-4.5,3.2,1.7,3],[2.8,-3.5,2.8,1.5,9],[.5,-8,3.6,2.2,5]]){const f=B({pos:[e,v.floor+.1,i],height:l,spread:1.6,size:o,count:24,speed:.02,wind:[.5,.1],seed:u,color:2384028,colorTop:4165584,opacity:.09});t.add(f),n.push(c=>f.userData.update(c))}const a=b({box:[-6,-2.2,-9,6,3.1,-.4],count:260,color:9095408,size:.03,opacity:.5,seed:8}),r=b({box:[-3,-1,-3.2,3.5,2.6,-.3],count:90,color:13627647,size:.024,opacity:.4,seed:21});for(const e of[a,r])t.add(e),n.push(i=>e.userData.update(i*3));return t}function H(s,n,{x0:t=-6.2,x1:a=6.2,z0:r=.18,z1:e=4.6,y:i=.014}={}){const l=new g({transparent:!0,depthWrite:!1,blending:W,fog:!1,polygonOffset:!0,polygonOffsetFactor:-2,polygonOffsetUnits:-2,uniforms:{uT:{value:0},uC:{value:new T(4892912)},uZ0:{value:r},uZ1:{value:e},uX0:{value:t},uX1:{value:a}},vertexShader:"varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }",fragmentShader:`
      uniform float uT, uZ0, uZ1, uX0, uX1; uniform vec3 uC; varying vec3 vW;
      ${A}
      void main(){
        float f = 1.0 - smoothstep(0.0, 1.0, (vW.z - uZ0) / (uZ1 - uZ0));
        float sx = smoothstep(uX0, uX0 + 1.5, vW.x) * (1.0 - smoothstep(uX1 - 1.5, uX1, vW.x));
        float c = caustic(vW.xz * 2.2, uT * 0.16) + 0.6 * caustic(vW.xz * 3.7 + 3.0, -uT * 0.12);
        float pools = 0.55 + 0.45 * sin(vW.x * 0.9 + 1.0) * sin(vW.x * 0.37 + 2.0);   /* the light comes through three panes, not evenly */
        gl_FragColor = vec4(uC * c * f * f * sx * pools * 1.3, 1.0);
      }`}),o=new w(new z(a-t,e-r),l);return o.rotation.x=-Math.PI/2,o.position.set((t+a)/2,i,(r+e)/2),o.renderOrder=3,s.add(h(o)),n.push(u=>{l.uniforms.uT.value=u}),o}function R(s,n){const t=new g({side:C,fog:!1,uniforms:{uT:{value:0}},vertexShader:"varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }",fragmentShader:`
      uniform float uT; varying vec3 vW;
      ${P}
      ${A}
      void main(){
        float d = distance(cameraPosition, vW);
        vec3 V = normalize(vW - cameraPosition);
        float swell = vn(vW.xz * 0.25 + uT * 0.02) * 0.6 + vn(vW.xz * 0.7 - uT * 0.03) * 0.4;
        float c = caustic(vW.xz * 0.9 + swell * 1.2, uT * 0.25);
        float c2 = caustic(vW.xz * 1.7 + 4.0 - swell, -uT * 0.2);
        float net = clamp(c * 0.9 + c2 * 0.5, 0.0, 1.4);
        vec3 deepc = vec3(0.012, 0.22, 0.62), lit = vec3(0.30, 0.86, 1.0);
        vec3 col = mix(deepc, lit, clamp(net * 0.7 + 0.14 * swell, 0.0, 1.0));
        float window = pow(max(-V.y, 0.0), 6.0);                     /* straight overhead is brightest (Snell's window) */
        col += vec3(0.1, 0.32, 0.42) * window;
        col = mix(col, vec3(0.03, 0.4, 0.85), 1.0 - exp(-d * 0.045));
        gl_FragColor = vec4(col, 1.0);
      }`}),a=new w(new z(20,17),t);return a.rotation.x=Math.PI/2,a.position.set(0,v.surf,-8.3),s.add(h(a)),n.push(r=>{t.uniforms.uT.value=r}),a}export{D as bubbleColumns,H as floorCaustics,k as godRays,G as silt,R as surfaceSheet};
