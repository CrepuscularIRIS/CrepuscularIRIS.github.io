import{W as G,H as _,x as E,y as Z,V as M,z as J,D as R,e as K,P as Y,F as f,I as k,J as Q,K as X,N as $}from"./index-BYAWy3sB.js";import{L as ee}from"./texlocal-BkU-iVjR.js";import"./react-three-fiber.esm-CLCXWAon.js";const te=`
  uniform mat4 textureMatrix;
  varying vec4 vUv; varying vec3 vWorld; varying vec2 vPlane;
  void main(){
    vPlane = uv;
    vUv = textureMatrix * vec4(position, 1.0);
    vWorld = (modelMatrix * vec4(position, 1.0)).xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`,ae=`
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
  }`;function ie({w:b,d:T,x:j=0,y:F=0,z:W=0,strength:S=.5,blur:D=1.2,f0:P=.08,tint:z=16777215,mask:h=null,maskRepeat:N=[1,1],size:v=768,fade:U=40,clipBias:V=.003,zones:H=0,smear:A=0,near:I=0}={}){const d=new G(v,Math.round(v*.5625),{type:_,samples:0}),s=new E({uniforms:{tDiffuse:{value:d.texture},textureMatrix:{value:new R},tMask:{value:h},uHasMask:{value:h?1:0},uMaskRepeat:{value:new M(...N)},uTint:{value:new J(z)},uStrength:{value:S},uBlur:{value:D},uF0:{value:P},uTexel:{value:new M(1/v,1/(v*.5625))},uFade:{value:U},uZones:{value:H},uSmear:{value:A},uNear:{value:I},uRain:{value:0},uRainT:{value:0},uRainScale:{value:4}},vertexShader:te,fragmentShader:ae,transparent:!0,blending:Z,depthWrite:!1,polygonOffset:!0,polygonOffsetFactor:-3,polygonOffsetUnits:-3}),t=new K(new Y(b,T),s);t.rotation.x=-Math.PI/2,t.position.set(j,F+.002,W),t.layers.set(ee),t.name="reflectFloor",t.userData.noOutline=!0;const i=new f,n=new f,g=new f,m=new f,w=new f,u=new k,p=new Q,r=new k,a=new X,x=new R;let c=!1;return t.onBeforeRender=(e,L,l)=>{if(c||s.uniforms.uStrength.value<=0)return;if(c=!0,i.setFromMatrixPosition(t.matrixWorld),n.set(0,0,1).transformDirection(t.matrixWorld),m.subVectors(i,l.position),m.dot(n)>0){c=!1;return}m.reflect(n).negate().add(i),x.extractRotation(l.matrixWorld),g.set(0,0,-1).applyMatrix4(x).add(l.position),w.subVectors(i,g).reflect(n).negate().add(i),a.position.copy(m),a.up.set(0,1,0).applyMatrix4(x).reflect(n),a.lookAt(w),a.far=l.far,a.updateMatrixWorld(),a.projectionMatrix.copy(l.projectionMatrix),a.layers.mask=l.layers.mask;const y=s.uniforms.textureMatrix.value;y.set(.5,0,0,.5,0,.5,0,.5,0,0,.5,.5,0,0,0,1),y.multiply(a.projectionMatrix).multiply(a.matrixWorldInverse).multiply(t.matrixWorld),p.setFromNormalAndCoplanarPoint(n,i).applyMatrix4(a.matrixWorldInverse),r.set(p.normal.x,p.normal.y,p.normal.z,p.constant);const o=a.projectionMatrix;u.x=(Math.sign(r.x)+o.elements[8])/o.elements[0],u.y=(Math.sign(r.y)+o.elements[9])/o.elements[5],u.z=-1,u.w=(1+o.elements[10])/o.elements[14],r.multiplyScalar(2/r.dot(u)),o.elements[2]=r.x,o.elements[6]=r.y,o.elements[10]=r.z+1-V,o.elements[14]=r.w;const O=e.getRenderTarget(),B=e.xr.enabled,C=e.shadowMap.autoUpdate,q=e.toneMapping;t.visible=!1,e.xr.enabled=!1,e.shadowMap.autoUpdate=!1,e.toneMapping=$,e.setRenderTarget(d),e.state.buffers.depth.setMask(!0),e.autoClear===!1&&e.clear(),e.render(L,a),e.toneMapping=q,e.xr.enabled=B,e.shadowMap.autoUpdate=C,e.setRenderTarget(O),t.visible=!0,c=!1},t.userData.dispose=()=>{d.dispose(),s.dispose(),t.geometry.dispose()},t.userData.rain={set(e){s.uniforms.uRain.value=e},tick(e){s.uniforms.uRainT.value=(s.uniforms.uRainT.value+e)%1e3}},t}export{ie as reflectFloor};
