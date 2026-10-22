import{W as L,H as _,x as z,y as E,V as h,z as G,D as F,e as q,P as J,F as p,I as W,J as K,K as Y,N as Q}from"./index-BTCq27pR.js";import{L as X}from"./texlocal-CGMucmHF.js";import"./react-three-fiber.esm-DLI5uoQM.js";const Z=`
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
  }`;function oe({w:P,d:T,x:k=0,y:b=0,z:R=0,strength:j=.5,blur:D=1.2,f0:U=.08,tint:V=16777215,mask:M=null,maskRepeat:S=[1,1],size:f=768,fade:A=40,clipBias:O=.003}={}){const x=new L(f,Math.round(f*.5625),{type:_,samples:0}),v=new z({uniforms:{tDiffuse:{value:x.texture},textureMatrix:{value:new F},tMask:{value:M},uHasMask:{value:M?1:0},uMaskRepeat:{value:new h(...S)},uTint:{value:new G(V)},uStrength:{value:j},uBlur:{value:D},uF0:{value:U},uTexel:{value:new h(1/f,1/(f*.5625))},uFade:{value:A}},vertexShader:Z,fragmentShader:$,transparent:!0,blending:E,depthWrite:!1,polygonOffset:!0,polygonOffsetFactor:-3,polygonOffsetUnits:-3}),t=new q(new J(P,T),v);t.rotation.x=-Math.PI/2,t.position.set(k,b+.002,R),t.layers.set(X),t.name="reflectFloor",t.userData.noOutline=!0;const n=new p,i=new p,g=new p,m=new p,y=new p,r=new W,u=new K,s=new W,a=new Y,d=new F;let c=!1;return t.onBeforeRender=(e,B,l)=>{if(c||v.uniforms.uStrength.value<=0)return;if(c=!0,n.setFromMatrixPosition(t.matrixWorld),i.set(0,0,1).transformDirection(t.matrixWorld),m.subVectors(n,l.position),m.dot(i)>0){c=!1;return}m.reflect(i).negate().add(n),d.extractRotation(l.matrixWorld),g.set(0,0,-1).applyMatrix4(d).add(l.position),y.subVectors(n,g).reflect(i).negate().add(n),a.position.copy(m),a.up.set(0,1,0).applyMatrix4(d).reflect(i),a.lookAt(y),a.far=l.far,a.updateMatrixWorld(),a.projectionMatrix.copy(l.projectionMatrix),a.layers.mask=l.layers.mask;const w=v.uniforms.textureMatrix.value;w.set(.5,0,0,.5,0,.5,0,.5,0,0,.5,.5,0,0,0,1),w.multiply(a.projectionMatrix).multiply(a.matrixWorldInverse).multiply(t.matrixWorld),u.setFromNormalAndCoplanarPoint(i,n).applyMatrix4(a.matrixWorldInverse),s.set(u.normal.x,u.normal.y,u.normal.z,u.constant);const o=a.projectionMatrix;r.x=(Math.sign(s.x)+o.elements[8])/o.elements[0],r.y=(Math.sign(s.y)+o.elements[9])/o.elements[5],r.z=-1,r.w=(1+o.elements[10])/o.elements[14],s.multiplyScalar(2/s.dot(r)),o.elements[2]=s.x,o.elements[6]=s.y,o.elements[10]=s.z+1-O,o.elements[14]=s.w;const C=e.getRenderTarget(),H=e.xr.enabled,I=e.shadowMap.autoUpdate,N=e.toneMapping;t.visible=!1,e.xr.enabled=!1,e.shadowMap.autoUpdate=!1,e.toneMapping=Q,e.setRenderTarget(x),e.state.buffers.depth.setMask(!0),e.autoClear===!1&&e.clear(),e.render(B,a),e.toneMapping=N,e.xr.enabled=H,e.shadowMap.autoUpdate=I,e.setRenderTarget(C),t.visible=!0,c=!1},t.userData.dispose=()=>{x.dispose(),v.dispose(),t.geometry.dispose()},t}export{oe as reflectFloor};
