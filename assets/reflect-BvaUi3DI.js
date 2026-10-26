import{W as E,H as G,x as Z,y as q,V as y,z as J,D as k,e as K,P as Y,F as f,I as b,J as Q,K as X,N as $}from"./index-BcYG4CQM.js";import{L as ee}from"./texlocal-Bz1tiBJZ.js";import"./react-three-fiber.esm-ym4uZNKT.js";const te=`
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
  varying vec4 vUv; varying vec3 vWorld; varying vec2 vPlane;
  void main(){
    vec2 uv = vUv.xy / vUv.w;
    float m = uHasMask > 0.5 ? texture2D(tMask, vPlane * uMaskRepeat).r : 1.0;
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
  }`;function ne({w:F,d:T,x:W=0,y:D=0,z:P=0,strength:R=.5,blur:S=1.2,f0:j=.08,tint:N=16777215,mask:g=null,maskRepeat:U=[1,1],size:m=768,fade:V=40,clipBias:A=.003,zones:O=0,smear:B=0,near:C=0}={}){const d=new E(m,Math.round(m*.5625),{type:G,samples:0}),p=new Z({uniforms:{tDiffuse:{value:d.texture},textureMatrix:{value:new k},tMask:{value:g},uHasMask:{value:g?1:0},uMaskRepeat:{value:new y(...U)},uTint:{value:new J(N)},uStrength:{value:R},uBlur:{value:S},uF0:{value:j},uTexel:{value:new y(1/m,1/(m*.5625))},uFade:{value:V},uZones:{value:O},uSmear:{value:B},uNear:{value:C}},vertexShader:te,fragmentShader:ae,transparent:!0,blending:q,depthWrite:!1,polygonOffset:!0,polygonOffsetFactor:-3,polygonOffsetUnits:-3}),t=new K(new Y(F,T),p);t.rotation.x=-Math.PI/2,t.position.set(W,D+.002,P),t.layers.set(ee),t.name="reflectFloor",t.userData.noOutline=!0;const r=new f,n=new f,M=new f,v=new f,h=new f,l=new b,u=new Q,s=new b,a=new X,x=new k;let c=!1;return t.onBeforeRender=(e,H,i)=>{if(c||p.uniforms.uStrength.value<=0)return;if(c=!0,r.setFromMatrixPosition(t.matrixWorld),n.set(0,0,1).transformDirection(t.matrixWorld),v.subVectors(r,i.position),v.dot(n)>0){c=!1;return}v.reflect(n).negate().add(r),x.extractRotation(i.matrixWorld),M.set(0,0,-1).applyMatrix4(x).add(i.position),h.subVectors(r,M).reflect(n).negate().add(r),a.position.copy(v),a.up.set(0,1,0).applyMatrix4(x).reflect(n),a.lookAt(h),a.far=i.far,a.updateMatrixWorld(),a.projectionMatrix.copy(i.projectionMatrix),a.layers.mask=i.layers.mask;const w=p.uniforms.textureMatrix.value;w.set(.5,0,0,.5,0,.5,0,.5,0,0,.5,.5,0,0,0,1),w.multiply(a.projectionMatrix).multiply(a.matrixWorldInverse).multiply(t.matrixWorld),u.setFromNormalAndCoplanarPoint(n,r).applyMatrix4(a.matrixWorldInverse),s.set(u.normal.x,u.normal.y,u.normal.z,u.constant);const o=a.projectionMatrix;l.x=(Math.sign(s.x)+o.elements[8])/o.elements[0],l.y=(Math.sign(s.y)+o.elements[9])/o.elements[5],l.z=-1,l.w=(1+o.elements[10])/o.elements[14],s.multiplyScalar(2/s.dot(l)),o.elements[2]=s.x,o.elements[6]=s.y,o.elements[10]=s.z+1-A,o.elements[14]=s.w;const I=e.getRenderTarget(),z=e.xr.enabled,L=e.shadowMap.autoUpdate,_=e.toneMapping;t.visible=!1,e.xr.enabled=!1,e.shadowMap.autoUpdate=!1,e.toneMapping=$,e.setRenderTarget(d),e.state.buffers.depth.setMask(!0),e.autoClear===!1&&e.clear(),e.render(H,a),e.toneMapping=_,e.xr.enabled=z,e.shadowMap.autoUpdate=L,e.setRenderTarget(I),t.visible=!0,c=!1},t.userData.dispose=()=>{d.dispose(),p.dispose(),t.geometry.dispose()},t}export{ne as reflectFloor};
