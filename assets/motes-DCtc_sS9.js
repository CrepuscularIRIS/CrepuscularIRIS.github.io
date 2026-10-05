import{t as S,u as l,d as w,A as g,C as y,V as r,w as A}from"./index-D-FHeTmP.js";import{r as B,L as z}from"./texlocal-CVHjPWQf.js";const F=`
  uniform float uTime, uSize, uScale;
  uniform vec3 uBox0, uBoxSize, uFocus; uniform float uFocusR;
  attribute vec4 aSeed;
  varying float vA;
  void main(){
    vec3 p = uBox0 + uBoxSize * aSeed.xyz;
    float t = uTime * (0.012 + aSeed.w * 0.016); /* slow drift: fast specks read as flicker */
    p += vec3(sin(t * 3.1 + aSeed.w * 40.0), sin(t * 2.3 + aSeed.x * 31.0) * 0.6 + t * 0.1, cos(t * 2.7 + aSeed.y * 27.0)) * 0.18;
    p = uBox0 + mod(p - uBox0, uBoxSize);
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    float tw = 0.8 + 0.2 * sin(uTime * 0.07 + aSeed.z * 20.0); /* no twinkle: brightness stays steady */
    float f = uFocusR > 0.0 ? 1.0 - smoothstep(0.0, uFocusR, distance(p, uFocus)) : 1.0;
    vA = tw * f * (0.4 + 0.6 * aSeed.w);
    gl_PointSize = clamp(uSize * uScale * (0.6 + aSeed.w * 0.8) / max(0.2, -mv.z), 1.0, 5.0);
  }`,h=`
  uniform vec3 uColor; uniform float uOpacity; varying float vA;
  void main(){
    float d = length(gl_PointCoord - 0.5) * 2.0;
    float a = smoothstep(1.0, 0.0, d); a *= a;
    gl_FragColor = vec4(uColor, a * vA * uOpacity);
  }`;function R({box:e,count:n=200,color:d=16769728,size:m=.03,opacity:c=.5,focus:o=null,seed:f=5,height:v=1080}={}){const p=B(f),s=new Float32Array(n*4);for(let t=0;t<s.length;t++)s[t]=p();const u=new S;u.setAttribute("position",new l(new Float32Array(n*3),3)),u.setAttribute("aSeed",new l(s,4));const i=new w({uniforms:{uTime:{value:0},uSize:{value:m},uScale:{value:v*.5},uBox0:{value:new r(e[0],e[1],e[2])},uBoxSize:{value:new r(e[3]-e[0],e[4]-e[1],e[5]-e[2])},uFocus:{value:new r(...o?o.pos:[0,0,0])},uFocusR:{value:o?o.radius:0},uColor:{value:new y(d)},uOpacity:{value:c}},vertexShader:F,fragmentShader:h,transparent:!0,depthWrite:!1,blending:g}),a=new A(u,i);return a.frustumCulled=!1,a.layers.set(z),a.name="motes",a.userData.update=t=>{i.uniforms.uTime.value=t},a.userData.dispose=()=>{u.dispose(),i.dispose()},a}export{R as m};
