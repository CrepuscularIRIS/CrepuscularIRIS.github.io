import{d as h,A as z,b as W,C as y,G as C,t as M,u as T,V as x,w as D,e as R,M as S,P as O,J as g}from"./index-VW0YcqsK.js";import{L as A}from"./texlocal-BnXnLrJd.js";import"./react-three-fiber.esm-DnmqodZ0.js";const U=`
  float h21(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
  float vn(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y); }
  /* ridged caustic network: two warped sine lattices multiplied, sharpened */
  float caustic(vec2 p, float t){
    vec2 w = p + 0.35 * vec2(sin(p.y * 1.7 + t * 0.9), cos(p.x * 1.3 - t * 0.8));
    float a = 1.0 - abs(sin(w.x * 2.3 + sin(w.y * 1.9 + t * 0.6) * 1.1));
    float b = 1.0 - abs(sin(w.y * 2.1 + sin(w.x * 2.4 - t * 0.5) * 1.2 + 1.7));
    float c = 1.0 - abs(sin((w.x + w.y) * 1.6 + sin(w.x * 3.1 - w.y * 2.2 + t * 0.7)));
    return pow(a * b, 3.0) * 1.6 + pow(b * c, 4.0) * 1.1 + pow(a * c, 5.0) * 0.8;
  }
`,E=`
  varying vec2 vUv; varying vec3 vN; varying vec3 vW;
  void main(){
    vUv = uv; vN = normalize(mat3(modelMatrix) * normal);
    vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz;
    gl_Position = projectionMatrix * viewMatrix * w;
  }`,V=`
  uniform vec3 uColor; uniform float uOp, uTime, uStreak, uEdge, uPow; varying vec2 vUv; varying vec3 vN; varying vec3 vW;
  ${U}
  void main(){
    float s = vn(vec2(vUv.x * uStreak, uTime * 0.02)) * 0.55 + vn(vec2(vUv.x * uStreak * 2.7 + 5.0, uTime * 0.035 + 3.0)) * 0.45;
    s = smoothstep(0.18, 0.95, s);
    float breathe = 0.78 + 0.22 * sin(uTime * 0.18 + vUv.x * 9.0);
    float along = pow(clamp(1.0 - vUv.y, 0.0, 1.0), uPow);
    float edge = uEdge > 0.5 ? smoothstep(0.0, 0.3, vUv.x) * smoothstep(1.0, 0.7, vUv.x) : 1.0;
    float view = abs(dot(normalize(vN), normalize(cameraPosition - vW)));
    float vis = mix(1.0, pow(view, 0.8), 0.85);
    float a = uOp * s * breathe * along * edge * vis;
    gl_FragColor = vec4(uColor * a, a);
  }`;function q({color:e=12577023,opacity:r=.2,streak:a=14,edge:o=!1,power:v=1.3}={}){return new h({uniforms:{uColor:{value:new y(e)},uOp:{value:r},uTime:{value:0},uStreak:{value:a},uEdge:{value:o?1:0},uPow:{value:v}},vertexShader:E,fragmentShader:V,transparent:!0,depthWrite:!1,side:W,blending:z,fog:!1})}function _(e,r){return e.traverse(a=>{a.layers.set(A),a.castShadow=!1,a.receiveShadow=!1,a.frustumCulled=!1}),e.renderOrder=6,e.userData.dynamic=!0,e.userData.update=a=>{for(const o of r)o.uniforms.uTime.value=a},e}function I({w0:e=.6,w1:r=1.8,len:a=6,x:o=0,y:v=0,z:u=0,tilt:l=0,yaw:c=0,mat:n}){const i=new M,s=[-e/2,0,0,e/2,0,0,r/2,-a,0,-r/2,-a,0];i.setAttribute("position",new g(s,3)),i.setAttribute("normal",new g([0,0,1,0,0,1,0,0,1,0,0,1],3)),i.setAttribute("uv",new g([0,0,1,0,1,1,0,1],2)),i.setIndex([0,2,1,0,3,2]);const f=new S(i,n);return f.position.set(o,v,u),f.rotation.set(0,c,l,"YXZ"),f}function H({w0:e,w1:r,len:a,x:o,y:v,z:u,tilt:l=0,mat:c,n=2}){const i=new C;for(let s=0;s<n;s++)i.add(I({w0:e,w1:r,len:a,x:o,y:v,z:u,tilt:l,yaw:s/n*Math.PI,mat:c}));return i}function L({pts:e,drop:r=3.4,shift:a=[0,0],grow:o=1.2,mat:v}){const u=e.length,l=e.reduce((t,p)=>t+p[0],0)/u,c=e.reduce((t,p)=>t+p[2],0)/u,n=[],i=[],s=[],f=[],d=e.map(([t,p,w])=>[l+(t-l)*o+a[0],p-r,c+(w-c)*o+a[1]]);for(let t=0;t<u;t++){const p=(t+1)%u,w=n.length/3;n.push(...e[t],...e[p],...d[p],...d[t]),i.push(t/u,0,(t+1)/u,0,(t+1)/u,1,t/u,1);const F=new x(...e[p]).sub(new x(...e[t])),B=new x(...d[t]).sub(new x(...e[t])),b=F.cross(B).normalize();for(let P=0;P<4;P++)s.push(b.x,b.y,b.z);f.push(w,w+2,w+1,w,w+3,w+2)}const m=new M;return m.setAttribute("position",new g(n,3)),m.setAttribute("normal",new g(s,3)),m.setAttribute("uv",new g(i,2)),m.setIndex(f),new S(m,v)}function $(e,r){const a=new C;a.name="godRays";for(const o of e)a.add(o);return _(a,r)}function Y({w:e,d:r,color:a=8376575,opacity:o=.5,scale:v=.9,speed:u=.16,vertical:l=!1,fadeY:c=null,x:n=0,y:i=0,z:s=0}){const f=new h({uniforms:{uColor:{value:new y(a)},uOp:{value:o},uTime:{value:0},uScale:{value:v},uSpeed:{value:u},uVert:{value:l?1:0},uFade:{value:new R(...c||[-1e3,1e3])}},vertexShader:"varying vec2 vUv; varying vec3 vW; void main(){ vUv = uv; vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }",fragmentShader:`
      uniform vec3 uColor; uniform float uOp, uTime, uScale, uSpeed, uVert; uniform vec2 uFade; varying vec2 vUv; varying vec3 vW;
      ${U}
      void main(){
        vec2 p = (uVert > 0.5 ? vec2(vW.x + vW.z, vW.y) : vW.xz) * uScale;
        float c = caustic(p, uTime * uSpeed * 6.0);
        float m = smoothstep(0.0, 0.5, vUv.x) * smoothstep(1.0, 0.5, vUv.x) * smoothstep(0.0, 0.5, vUv.y) * smoothstep(1.0, 0.5, vUv.y);
        float fy = uVert > 0.5 ? smoothstep(uFade.x, uFade.y, vW.y) : 1.0;
        float a = c * m * fy * uOp;
        gl_FragColor = vec4(uColor * a, a);
      }`,transparent:!0,depthWrite:!1,side:W,blending:z,fog:!1}),d=new S(new O(e,r),f);return l||(d.rotation.x=-Math.PI/2),d.position.set(n,i,s),_(d,[f])}function j({w:e,d:r,x:a=0,y:o=0,z:v=0,deep:u=2783432,bright:l=12118271}){const c=new h({uniforms:{uDeep:{value:new y(u)},uBright:{value:new y(l)},uTime:{value:0}},vertexShader:"varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }",fragmentShader:`
      uniform vec3 uDeep, uBright; uniform float uTime; varying vec3 vW;
      ${U}
      void main(){
        vec2 p = vW.xz * 0.55;
        float c = caustic(p, uTime * 0.7) * 0.8 + caustic(p * 2.1 + 3.0, uTime * 0.9) * 0.35;
        float swell = vn(p * 0.8 + uTime * 0.03);
        vec3 col = mix(uDeep, uBright, clamp(0.28 + swell * 0.5 + c * 0.5, 0.0, 1.0));
        col += vec3(0.6, 0.9, 1.0) * pow(clamp(c, 0.0, 1.5), 2.0) * 0.25;
        gl_FragColor = vec4(col, 1.0);
      }`,fog:!1}),n=new S(new O(e,r),c);return n.rotation.x=Math.PI/2,n.position.set(a,o,v),n.layers.set(A),n.userData.dynamic=!0,n.userData.update=i=>{c.uniforms.uTime.value=i},n}function J({pos:e=[0,0,0],height:r=4,radius:a=.12,count:o=36,size:v=.05,opacity:u=.7,speed:l=.07,seed:c=3,viewH:n=1080}){const i=(()=>{let t=c>>>0||1;return()=>(t=t*1664525+1013904223>>>0)/4294967296})(),s=new Float32Array(o*4);for(let t=0;t<s.length;t++)s[t]=i();const f=new M;f.setAttribute("position",new T(new Float32Array(o*3),3)),f.setAttribute("aSeed",new T(s,4));const d=new h({uniforms:{uTime:{value:0},uPos:{value:new x(...e)},uH:{value:r},uR:{value:a},uSize:{value:v},uSpeed:{value:l},uOp:{value:u},uScale:{value:n*.5}},vertexShader:`
      uniform float uTime, uH, uR, uSize, uSpeed, uScale; uniform vec3 uPos; attribute vec4 aSeed; varying float vA; varying float vU;
      void main(){
        float u = fract(aSeed.x + uTime * uSpeed * (0.7 + aSeed.w * 0.6));
        vec3 p = uPos;
        p.y += u * uH;
        float sway = sin(u * 11.0 + aSeed.y * 6.283) * 0.5 + sin(u * 5.0 + aSeed.z * 6.283) * 0.5;
        p.x += (aSeed.y - 0.5) * 2.0 * uR * (0.4 + u) + sway * uR * 0.4;
        p.z += (aSeed.z - 0.5) * 2.0 * uR * (0.4 + u) + sway * uR * 0.3;
        vec4 mv = viewMatrix * modelMatrix * vec4(p, 1.0);
        gl_Position = projectionMatrix * mv;
        float s = uSize * (0.5 + aSeed.w) * (0.7 + 0.8 * u);
        gl_PointSize = clamp(s * uScale * 2.0 / max(0.2, -mv.z), 1.5, 14.0);
        vA = smoothstep(0.0, 0.06, u) * (1.0 - smoothstep(0.9, 1.0, u));
        vU = u;
      }`,fragmentShader:`
      uniform float uOp; varying float vA; varying float vU;
      void main(){
        vec2 q = gl_PointCoord - 0.5; float d = length(q) * 2.0;
        if (d > 1.0) discard;
        float rim = smoothstep(0.55, 0.95, d) * (1.0 - smoothstep(0.95, 1.0, d));
        float hi = smoothstep(0.35, 0.0, length(q - vec2(-0.14, 0.14)));
        float a = (rim * 0.75 + hi * 0.9 + 0.08) * vA * uOp;
        gl_FragColor = vec4(vec3(0.75, 0.92, 1.0) * a, a);
      }`,transparent:!0,depthWrite:!1,blending:z,fog:!1}),m=new D(f,d);return m.frustumCulled=!1,m.layers.set(A),m.renderOrder=7,m.name="bubbleColumn",m.userData.dynamic=!0,m.userData.update=t=>{d.uniforms.uTime.value=t},m}export{q as beamMat,H as beamStar,J as bubbleColumn,Y as causticPlane,$ as godRays,L as prismBeam,I as slabBeam,j as surfaceUnderside};
