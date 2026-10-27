import{Z as F,ad as b,x as h,F as V,ah as _,z as T,y as W,e as w,P as z,U as k,X as C}from"./index-BYAWy3sB.js";import{g as x,a as M,n as y}from"./texlocal-BkU-iVjR.js";import{s as O}from"./smoke-DC1YGJmE.js";import{m as S}from"./motes-Bej4vtYI.js";import{W as l}from"./uw-6xjGp55L.js";import"./react-three-fiber.esm-CLCXWAon.js";const P=`
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
`,X=(n,r)=>new h({transparent:!0,depthWrite:!1,blending:W,side:C,fog:!1,uniforms:{uT:{value:0},uColor:{value:new T(n)},uI:{value:r},uSeed:{value:0},uTop:{value:l.surf},uBot:{value:l.floor}},vertexShader:"varying vec3 vW; varying vec3 vN; varying vec2 vUv; void main(){ vUv = uv; vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; vN = normalize(mat3(modelMatrix) * normal); gl_Position = projectionMatrix * viewMatrix * w; }",fragmentShader:`
    uniform float uT, uI, uSeed, uTop, uBot; uniform vec3 uColor; varying vec3 vW; varying vec3 vN; varying vec2 vUv;
    ${P}
    void main(){
      vec3 V = normalize(cameraPosition - vW);
      float facing = abs(dot(normalize(vN), V));
      float edge = pow(facing, 1.6);                              /* the cone fades toward its silhouette: reads as a volume */
      float v = clamp((vW.y - uBot) / (uTop - uBot), 0.0, 1.0);    /* 1 at the surface */
      float vert = pow(v, 1.5) * smoothstep(0.0, 0.22, 1.0 - v);   /* strongest a little under the surface (a slow top fade: no stripe along the surface line), gone at the floor */
      float band = 0.55 + 0.45 * vn(vec2(vUv.x * 6.0 + uSeed, v * 5.0 - uT * 0.05));      /* drifting light and dark bands along the shaft */
      float pulse = 0.7 + 0.3 * sin(uT * 0.18 + uSeed * 4.0);
      float dist = distance(cameraPosition, vW);
      float near = smoothstep(0.4, 2.5, dist);   /* no hard slab when the camera nears a shaft */
      float far = mix(1.0, 0.2, smoothstep(7.0, 15.0, dist));   /* the water swallows the far shafts: no pale band stacked on the back wall */
      gl_FragColor = vec4(uColor * edge * vert * band * pulse * uI * near * far, 1.0);
    }`});function N(n,r,{count:t=16,seed:a=7}={}){const s=x(n,"godRays"),e=M(a),i=l.surf-l.floor+.6,c=[];for(let o=0;o<t;o++){const u=.18+e()*.42,f=u*(1.8+e()*1.2),v=X(o%3===0?12906213:8571078,.16+e()*.12);v.uniforms.uSeed.value=e()*20,c.push(v);const d=new w(new k(u,f,i,24,1,!0),v),m=-7+o/(t-1)*14+(e()-.5)*1.4,p=-1.5-e()*11,g=.14+e()*.06;d.position.set(m-Math.sin(g)*i*.5,l.surf-i*.5*Math.cos(g),p+i*.12),d.rotation.set(-.22,0,g),d.renderOrder=4,d.frustumCulled=!1,s.add(y(d))}return r.push(o=>{for(const u of c)u.uniforms.uT.value=o}),s}function D(n,r,t,{floorY:a=l.floor+.12}={}){const s=x(n,"bubbleColumns"),e=M(31);for(const[i,c,o,u]of t){const f=new F,v=new Float32Array(o*4);for(let p=0;p<v.length;p++)v[p]=e();f.setAttribute("position",new b(new Float32Array(o*3),3)),f.setAttribute("aSeed",new b(v,4));const d=new h({transparent:!0,depthWrite:!1,fog:!1,uniforms:{uT:{value:0},uV:{value:new V(i,a,c)},uTop:{value:l.surf-.05},uSize:{value:u},uScale:{value:540}},vertexShader:`
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
        }`}),m=new _(f,d);m.frustumCulled=!1,m.renderOrder=6,s.add(y(m)),r.push(p=>{d.uniforms.uT.value=p})}return s}function G(n,r){const t=x(n,"silt");for(const[e,i,c,o,u]of[[-3,-4.5,3.2,1.7,3],[2.8,-3.5,2.8,1.5,9],[.5,-8,3.6,2.2,5]]){const f=O({pos:[e,l.floor+.1,i],height:c,spread:1.6,size:o,count:24,speed:.02,wind:[.5,.1],seed:u,color:1853004,colorTop:3764080,opacity:.065});t.add(f),r.push(v=>f.userData.update(v))}const a=S({box:[-6,-2.2,-9,6,3.1,-.4],count:260,color:8635581,size:.03,opacity:.42,seed:8}),s=S({box:[-3,-1,-3.2,3.5,2.6,-.3],count:90,color:13166815,size:.024,opacity:.36,seed:21});for(const e of[a,s])t.add(e),r.push(i=>e.userData.update(i*3));return t}function H(n,r,{x0:t=-6.2,x1:a=6.2,z0:s=.18,z1:e=3.4,y:i=.014}={}){const c=new h({transparent:!0,depthWrite:!1,blending:W,fog:!1,polygonOffset:!0,polygonOffsetFactor:-2,polygonOffsetUnits:-2,uniforms:{uT:{value:0},uC:{value:new T(6473924)},uZ0:{value:s},uZ1:{value:e},uX0:{value:t},uX1:{value:a}},vertexShader:"varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }",fragmentShader:`
      uniform float uT, uZ0, uZ1, uX0, uX1; uniform vec3 uC; varying vec3 vW;
      ${A}
      void main(){
        float f = 1.0 - smoothstep(0.0, 1.0, (vW.z - uZ0) / (uZ1 - uZ0));
        float sx = smoothstep(uX0, uX0 + 1.5, vW.x) * (1.0 - smoothstep(uX1 - 1.5, uX1, vW.x));
        float c = caustic(vW.xz * 2.2, uT * 0.16) + 0.6 * caustic(vW.xz * 3.7 + 3.0, -uT * 0.12);
        float pools = 0.55 + 0.45 * sin(vW.x * 0.9 + 1.0) * sin(vW.x * 0.37 + 2.0);   /* the light comes through three panes, not evenly */
        gl_FragColor = vec4(uC * c * f * f * sx * pools * 0.42, 1.0);
      }`}),o=new w(new z(a-t,e-s),c);return o.rotation.x=-Math.PI/2,o.position.set((t+a)/2,i,(s+e)/2),o.renderOrder=3,n.add(y(o)),r.push(u=>{c.uniforms.uT.value=u}),o}function R(n,r){const t=new h({side:C,fog:!1,uniforms:{uT:{value:0}},vertexShader:"varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }",fragmentShader:`
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
        vec3 deepc = vec3(0.012, 0.16, 0.19), lit = vec3(0.28, 0.72, 0.70);
        vec3 col = mix(deepc, lit, clamp(net * 0.7 + 0.14 * swell, 0.0, 1.0));
        float window = pow(max(-V.y, 0.0), 6.0);                     /* straight overhead is brightest (Snell's window) */
        col += vec3(0.08, 0.26, 0.27) * window;
        col = mix(col, vec3(0.02, 0.22, 0.29), 1.0 - exp(-d * 0.075));   /* the far edge melts into the water just below it, not into a brighter band */
        /* toward the back and side walls the sheet sinks into the same dark water as the rock under its edge: a long
           ramp (7 m), so the far surface never reads as a pale strip with a level seam against the walls */
        float e = min(vW.z - (${l.back.toFixed(1)}), ${l.side.toFixed(1)} - abs(vW.x));
        col = mix(vec3(0.002, 0.02, 0.026), col, smoothstep(0.0, 7.0, e));
        gl_FragColor = vec4(col, 1.0);
      }`});t.polygonOffset=!0;const a=new w(new z(20,17),t);return a.rotation.x=Math.PI/2,a.position.set(0,l.surf,-8.3),a.castShadow=!1,n.add(a),r.push(s=>{t.uniforms.uT.value=s}),a}export{A as CAUSTIC,D as bubbleColumns,H as floorCaustics,N as godRays,G as silt,R as surfaceSheet};
