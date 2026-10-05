import{W as L,c as _,d as E,A as G,e as h,C as z,f as F,M as q,P as Y,V as p,g as W,h as J,i as K,N as Q}from"./index-dtWvHS9r.js";import{L as X}from"./texlocal-5yqF4Mqj.js";const Z=`
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
  }`;function ae({w:P,d:T,x:k=0,y:b=0,z:R=0,strength:j=.5,blur:U=1.2,f0:V=.08,tint:D=16777215,mask:M=null,maskRepeat:S=[1,1],size:f=768,fade:A=40,clipBias:C=.003}={}){const d=new L(f,Math.round(f*.5625),{type:_,samples:0}),v=new E({uniforms:{tDiffuse:{value:d.texture},textureMatrix:{value:new F},tMask:{value:M},uHasMask:{value:M?1:0},uMaskRepeat:{value:new h(...S)},uTint:{value:new z(D)},uStrength:{value:j},uBlur:{value:U},uF0:{value:V},uTexel:{value:new h(1/f,1/(f*.5625))},uFade:{value:A}},vertexShader:Z,fragmentShader:$,transparent:!0,blending:G,depthWrite:!1,polygonOffset:!0,polygonOffsetFactor:-3,polygonOffsetUnits:-3}),t=new q(new Y(P,T),v);t.rotation.x=-Math.PI/2,t.position.set(k,b+.002,R),t.layers.set(X),t.name="reflectFloor",t.userData.noOutline=!0;const n=new p,i=new p,g=new p,c=new p,w=new p,r=new W,u=new J,s=new W,a=new K,x=new F;let m=!1;return t.onBeforeRender=(e,O,l)=>{if(m||v.uniforms.uStrength.value<=0)return;if(m=!0,n.setFromMatrixPosition(t.matrixWorld),i.set(0,0,1).transformDirection(t.matrixWorld),c.subVectors(n,l.position),c.dot(i)>0){m=!1;return}c.reflect(i).negate().add(n),x.extractRotation(l.matrixWorld),g.set(0,0,-1).applyMatrix4(x).add(l.position),w.subVectors(n,g).reflect(i).negate().add(n),a.position.copy(c),a.up.set(0,1,0).applyMatrix4(x).reflect(i),a.lookAt(w),a.far=l.far,a.updateMatrixWorld(),a.projectionMatrix.copy(l.projectionMatrix),a.layers.mask=l.layers.mask;const y=v.uniforms.textureMatrix.value;y.set(.5,0,0,.5,0,.5,0,.5,0,0,.5,.5,0,0,0,1),y.multiply(a.projectionMatrix).multiply(a.matrixWorldInverse).multiply(t.matrixWorld),u.setFromNormalAndCoplanarPoint(i,n).applyMatrix4(a.matrixWorldInverse),s.set(u.normal.x,u.normal.y,u.normal.z,u.constant);const o=a.projectionMatrix;r.x=(Math.sign(s.x)+o.elements[8])/o.elements[0],r.y=(Math.sign(s.y)+o.elements[9])/o.elements[5],r.z=-1,r.w=(1+o.elements[10])/o.elements[14],s.multiplyScalar(2/s.dot(r)),o.elements[2]=s.x,o.elements[6]=s.y,o.elements[10]=s.z+1-C,o.elements[14]=s.w;const B=e.getRenderTarget(),N=e.xr.enabled,H=e.shadowMap.autoUpdate,I=e.toneMapping;t.visible=!1,e.xr.enabled=!1,e.shadowMap.autoUpdate=!1,e.toneMapping=Q,e.setRenderTarget(d),e.state.buffers.depth.setMask(!0),e.autoClear===!1&&e.clear(),e.render(O,a),e.toneMapping=I,e.xr.enabled=N,e.shadowMap.autoUpdate=H,e.setRenderTarget(B),t.visible=!0,m=!1},t.userData.dispose=()=>{d.dispose(),v.dispose(),t.geometry.dispose()},t}export{ae as r};
