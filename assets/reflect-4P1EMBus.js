import{W as L,i as _,S as E,k as G,V as h,C as q,l as F,f as z,g as Y,m as p,n as W,q as J,t as K,N as Q}from"./index-D0AdTg2o.js";import{L as X}from"./texlocal-B1Iq89O3.js";import"./react-three-fiber.esm-lrwiZj00.js";import"./GLTFLoader-BsrA9sm_.js";const Z=`
  uniform mat4 textureMatrix;
  varying vec4 vUv; varying vec3 vWorld; varying vec2 vPlane;
  void main(){
    vPlane = uv;
    vUv = textureMatrix * vec4(position, 1.0);
    vWorld = (modelMatrix * vec4(position, 1.0)).xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`,$=`
  uniform sampler2D tDiffuse, tMask;
  uniform vec3 uTint; uniform float uStrength, uBlur, uHasMask, uF0, uFade;
  uniform vec2 uTexel, uMaskRepeat;
  varying vec4 vUv; varying vec3 vWorld; varying vec2 vPlane;
  void main(){
    vec2 uv = vUv.xy / vUv.w;
    vec3 c = vec3(0.0);
    float tw = 0.0;
    for (int i = -2; i <= 2; i++) for (int j = -2; j <= 2; j++) {
      float w = exp(-float(i*i + j*j) * 0.35);
      c += texture2D(tDiffuse, uv + vec2(float(i), float(j)) * uTexel * uBlur).rgb * w; tw += w;
    }
    c /= tw;
    vec3 V = normalize(cameraPosition - vWorld);
    float fres = uF0 + (1.0 - uF0) * pow(1.0 - clamp(V.y, 0.0, 1.0), 4.0);
    float m = uHasMask > 0.5 ? texture2D(tMask, vPlane * uMaskRepeat).r : 1.0;
    float dist = distance(cameraPosition, vWorld);
    float fade = 1.0 - smoothstep(uFade * 0.5, uFade, dist);
    gl_FragColor = vec4(c * uTint * (uStrength * fres * m * fade), 1.0);
  }`;function st({w:k,d:T,x:P=0,y:b=0,z:R=0,strength:j=.5,blur:U=1.2,f0:V=.08,tint:D=16777215,mask:M=null,maskRepeat:S=[1,1],size:f=768,fade:A=40,clipBias:C=.003}={}){const d=new L(f,Math.round(f*.5625),{type:_,samples:0}),v=new E({uniforms:{tDiffuse:{value:d.texture},textureMatrix:{value:new F},tMask:{value:M},uHasMask:{value:M?1:0},uMaskRepeat:{value:new h(...S)},uTint:{value:new q(D)},uStrength:{value:j},uBlur:{value:U},uF0:{value:V},uTexel:{value:new h(1/f,1/(f*.5625))},uFade:{value:A}},vertexShader:Z,fragmentShader:$,transparent:!0,blending:G,depthWrite:!1,polygonOffset:!0,polygonOffsetFactor:-3,polygonOffsetUnits:-3}),e=new z(new Y(k,T),v);e.rotation.x=-Math.PI/2,e.position.set(P,b+.002,R),e.layers.set(X),e.name="reflectFloor",e.userData.noOutline=!0;const n=new p,i=new p,g=new p,m=new p,w=new p,r=new W,u=new J,s=new W,a=new K,x=new F;let c=!1;return e.onBeforeRender=(t,O,l)=>{if(c||v.uniforms.uStrength.value<=0)return;if(c=!0,n.setFromMatrixPosition(e.matrixWorld),i.set(0,0,1).transformDirection(e.matrixWorld),m.subVectors(n,l.position),m.dot(i)>0){c=!1;return}m.reflect(i).negate().add(n),x.extractRotation(l.matrixWorld),g.set(0,0,-1).applyMatrix4(x).add(l.position),w.subVectors(n,g).reflect(i).negate().add(n),a.position.copy(m),a.up.set(0,1,0).applyMatrix4(x).reflect(i),a.lookAt(w),a.far=l.far,a.updateMatrixWorld(),a.projectionMatrix.copy(l.projectionMatrix),a.layers.mask=l.layers.mask;const y=v.uniforms.textureMatrix.value;y.set(.5,0,0,.5,0,.5,0,.5,0,0,.5,.5,0,0,0,1),y.multiply(a.projectionMatrix).multiply(a.matrixWorldInverse).multiply(e.matrixWorld),u.setFromNormalAndCoplanarPoint(i,n).applyMatrix4(a.matrixWorldInverse),s.set(u.normal.x,u.normal.y,u.normal.z,u.constant);const o=a.projectionMatrix;r.x=(Math.sign(s.x)+o.elements[8])/o.elements[0],r.y=(Math.sign(s.y)+o.elements[9])/o.elements[5],r.z=-1,r.w=(1+o.elements[10])/o.elements[14],s.multiplyScalar(2/s.dot(r)),o.elements[2]=s.x,o.elements[6]=s.y,o.elements[10]=s.z+1-C,o.elements[14]=s.w;const B=t.getRenderTarget(),N=t.xr.enabled,H=t.shadowMap.autoUpdate,I=t.toneMapping;e.visible=!1,t.xr.enabled=!1,t.shadowMap.autoUpdate=!1,t.toneMapping=Q,t.setRenderTarget(d),t.state.buffers.depth.setMask(!0),t.autoClear===!1&&t.clear(),t.render(O,a),t.toneMapping=I,t.xr.enabled=N,t.shadowMap.autoUpdate=H,t.setRenderTarget(B),e.visible=!0,c=!1},e.userData.dispose=()=>{d.dispose(),v.dispose(),e.geometry.dispose()},e}export{st as reflectFloor};
