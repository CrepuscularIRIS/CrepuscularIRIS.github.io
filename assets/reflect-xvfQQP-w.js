import{W as _,H as Z,x as K,y as J,V as k,z as Y,D as T,e as Q,P as X,F as c,I as j,J as $,K as ee,N as te}from"./index-D4ZUHyJy.js";import{L as ae}from"./texlocal-BOxE6shg.js";import"./react-three-fiber.esm-CGynnRJ9.js";const oe=`
  uniform mat4 textureMatrix;
  varying vec4 vUv; varying vec3 vWorld; varying vec2 vPlane;
  void main(){
    vPlane = uv;
    vUv = textureMatrix * vec4(position, 1.0);
    vWorld = (modelMatrix * vec4(position, 1.0)).xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`,re=`
  uniform sampler2D tDiffuse, tMask;
  uniform vec3 uTint; uniform float uStrength, uBlur, uHasMask, uF0, uFade, uZones, uSmear, uNear;
  uniform vec2 uTexel, uMaskRepeat;
  uniform float uRain, uRainT, uRainScale;
  varying vec4 vUv; varying vec3 vWorld; varying vec2 vPlane;
  /* falling-drop rings: per cell of a 3x3 neighbourhood, a ring born at a hashed point and time, expanding and fading;
     returns the rings' slope (threejs-conference rainRipples.js, MIT). Reversed smoothstep edges are written as
     1 - smoothstep(ordered) (GLSL leaves edge0 >= edge1 undefined); normalize(0) is guarded. */
  float rHash12(vec2 p){ vec3 p3 = fract(p.xyx * 0.1031); p3 += dot(p3, p3.yzx + 19.19); return fract((p3.x + p3.y) * p3.z); }
  vec2 rHash22(vec2 p){ vec3 p3 = fract(p.xyx * vec3(0.1031, 0.1030, 0.0973)); p3 += dot(p3, p3.yzx + 19.19); return fract((p3.xx + p3.yz) * p3.zy); }
  float rRing(float d){ return sin(31.0 * d) * smoothstep(-0.6, -0.3, d) * (1.0 - smoothstep(-0.3, 0.0, d)); }
  vec2 rainRipples(vec2 q, float time){
    vec2 p0 = floor(q), circles = vec2(0.0);
    for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
      vec2 pi = p0 + vec2(float(i), float(j));
      vec2 v = pi + rHash22(pi) - q;
      float t = fract(0.3 * time + rHash12(pi));
      float d = length(v) - 2.0 * t;
      float der = (rRing(d + 0.001) - rRing(d - 0.001)) / 0.002 * (1.0 - t) * (1.0 - t);
      circles += v / max(length(v), 1e-4) * der * 0.5;
    }
    return circles / 9.0;
  }
  void main(){
    vec2 uv = vUv.xy / vUv.w;
    float m = uHasMask > 0.5 ? texture2D(tMask, vPlane * uMaskRepeat).r : 1.0;
    /* rain breaks the mirror where it is wet; the mask is read where it always was. Rings a few cm apart turn to grain
       on the mid-ground (a few pixels each): they fade from 6 m and are gone by 15 m */
    if (uRain > 0.0) uv += rainRipples(vWorld.xz * uRainScale, uRainT * 3.0) * (uRain * 0.03 * smoothstep(0.08, 0.2, m) * (1.0 - smoothstep(6.0, 15.0, distance(cameraPosition, vWorld))));
    /* zones (0..1): the blur follows the wetness, standing water sharp, damp asphalt rough */
    float blur = uBlur * mix(1.0, mix(2.6, 0.55, smoothstep(0.25, 0.85, m)), uZones);
    vec3 c = vec3(0.0);
    float tw = 0.0;
    for (int i = -2; i <= 2; i++) for (int j = -2; j <= 2; j++) {
      float w = exp(-float(i*i + j*j) * 0.35);
      c += texture2D(tDiffuse, uv + vec2(float(i), float(j)) * uTexel * blur).rgb * w; tw += w;
    }
    c /= tw;
    vec3 V = normalize(cameraPosition - vWorld);
    float fres = uF0 + (1.0 - uF0) * pow(1.0 - clamp(V.y, 0.0, 1.0), 4.0);
    float dist = distance(cameraPosition, vWorld);
    float fade = 1.0 - smoothstep(uFade * 0.5, uFade, dist);
    /* near (metres): the reflection eases in from the visitor's feet instead of starting at full strength */
    if (uNear > 0.0) fade *= mix(0.45, 1.0, smoothstep(uNear * 0.3, uNear, dist));
    /* smear: lights stretch into vertical streaks on wet asphalt; added over the mirror where it is merely damp,
       so puddles stay mirrors and the rest breaks into streaks instead of one even sheen */
    vec3 s = vec3(0.0);
    if (uSmear > 0.0) {
      for (int k = 1; k <= 8; k++) { float o = float(k) * 7.0; s += texture2D(tDiffuse, uv + vec2(0.0, o) * uTexel).rgb + texture2D(tDiffuse, uv - vec2(0.0, o) * uTexel).rgb; }
      s *= uSmear / 16.0 * (1.0 - smoothstep(0.55, 0.95, m)) * smoothstep(0.08, 0.35, m);
    }
    gl_FragColor = vec4((c * m + s) * uTint * (uStrength * fres * fade), 1.0);
  }`;function le({w:S,d:F,x:W=0,y:D=0,z:P=0,strength:O=.5,blur:z=1.2,f0:N=.08,tint:U=16777215,mask:w=null,maskRepeat:V=[1,1],size:d=768,fade:H=40,clipBias:L=.003,zones:A=0,smear:B=0,near:I=0}={}){const g=new _(d,Math.round(d*.5625),{type:Z,samples:0}),r=new K({uniforms:{tDiffuse:{value:g.texture},textureMatrix:{value:new T},tMask:{value:w},uHasMask:{value:w?1:0},uMaskRepeat:{value:new k(...V)},uTint:{value:new Y(U)},uStrength:{value:O},uBlur:{value:z},uF0:{value:N},uTexel:{value:new k(1/d,1/(d*.5625))},uFade:{value:H},uZones:{value:A},uSmear:{value:B},uNear:{value:I},uRain:{value:0},uRainT:{value:0},uRainScale:{value:4}},vertexShader:oe,fragmentShader:re,transparent:!0,blending:J,depthWrite:!1,polygonOffset:!0,polygonOffsetFactor:-3,polygonOffsetUnits:-3}),t=new Q(new X(S,F),r);t.rotation.x=-Math.PI/2,t.position.set(W,D+.002,P),t.layers.set(ae),t.name="reflectFloor",t.userData.noOutline=!0;const l=new c,u=new c,M=new c,x=new c,R=new c,p=new j,m=new $,n=new j,o=new ee,y=new T;let h=!1;t.onBeforeRender=(e,v,a)=>{if(h||r.uniforms.uStrength.value<=0)return;if(h=!0,l.setFromMatrixPosition(t.matrixWorld),u.set(0,0,1).transformDirection(t.matrixWorld),x.subVectors(l,a.position),x.dot(u)>0){h=!1;return}x.reflect(u).negate().add(l),y.extractRotation(a.matrixWorld),M.set(0,0,-1).applyMatrix4(y).add(a.position),R.subVectors(l,M).reflect(u).negate().add(l),o.position.copy(x),o.up.set(0,1,0).applyMatrix4(y).reflect(u),o.lookAt(R),o.far=a.far,o.updateMatrixWorld(),o.projectionMatrix.copy(a.projectionMatrix),o.layers.mask=a.layers.mask;const i=r.uniforms.textureMatrix.value;i.set(.5,0,0,.5,0,.5,0,.5,0,0,.5,.5,0,0,0,1),i.multiply(o.projectionMatrix).multiply(o.matrixWorldInverse).multiply(t.matrixWorld),m.setFromNormalAndCoplanarPoint(u,l).applyMatrix4(o.matrixWorldInverse),n.set(m.normal.x,m.normal.y,m.normal.z,m.constant);const s=o.projectionMatrix;p.x=(Math.sign(n.x)+s.elements[8])/s.elements[0],p.y=(Math.sign(n.y)+s.elements[9])/s.elements[5],p.z=-1,p.w=(1+s.elements[10])/s.elements[14],n.multiplyScalar(2/n.dot(p)),s.elements[2]=n.x,s.elements[6]=n.y,s.elements[10]=n.z+1-L,s.elements[14]=n.w;const C=e.getRenderTarget(),q=e.xr.enabled,E=e.shadowMap.autoUpdate,G=e.toneMapping;t.visible=!1,e.xr.enabled=!1,e.shadowMap.autoUpdate=!1,e.toneMapping=te,e.setRenderTarget(g),e.state.buffers.depth.setMask(!0),e.autoClear===!1&&e.clear(),e.render(v,o),e.toneMapping=G,e.xr.enabled=q,e.shadowMap.autoUpdate=E,e.setRenderTarget(C),t.visible=!0,h=!1},t.userData.dispose=()=>{g.dispose(),r.dispose(),t.geometry.dispose()};const b={smear:"uSmear",strength:"uStrength",blur:"uBlur"};let f=null;return t.userData.rain={set(e,v=null){if(r.uniforms.uRain.value=e,e>0&&v&&!f){f=Object.fromEntries(Object.values(b).map(a=>[a,r.uniforms[a].value]));for(const[a,i]of Object.entries(b))v[a]!=null&&(r.uniforms[i].value=f[i]*v[a])}else if(!(e>0)&&f){for(const[a,i]of Object.entries(f))r.uniforms[a].value=i;f=null}},tick(e){r.uniforms.uRainT.value=(r.uniforms.uRainT.value+e)%1e3}},t}export{le as reflectFloor};
