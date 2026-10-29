import{al as U,P as q,z as f,$,I as j,F as Y,e as K}from"./index-DtL6CxF2.js";import{g as z,a as Z,k as N,x as R,P as A,a7 as x,h as J,n as Q,a8 as P,U as ee}from"./texlocal-LTzE_-Aj.js";import{BC_NOISE as te,BC_GLSL as ae}from"./cloudlayer-D31Kd841.js";import{CAUSTIC as oe}from"./atmos-BT_M7i45.js";import"./react-three-fiber.esm-DirPJ1ic.js";import"./smoke-BuV3jIgh.js";import"./motes-C1g7tfjQ.js";import"./uw-BgXmyLKc.js";const g=[[40,9],[8,3.8],[-3,2.45],[-5.6,2.05],[-10,.4],[-20,-2.2],[-40,-6.8],[-70,-13.5],[-112,-22.9],[-200,-38],[-400,-60]];function L(r){for(let t=0;t<g.length-1;t++){const[a,o]=g[t],[c,n]=g[t+1];if(r<=a&&r>=c)return P(o,n,(a-r)/(a-c))}return r>g[0][0]?g[0][1]:g[g.length-1][1]}const re=`
  uniform vec4 uSwash;
  float swH(float n){ return fract(sin(n * 127.1) * 43758.5453); }
  float swN(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    float a = swH(i.x + i.y * 57.0), b = swH(i.x + 1.0 + i.y * 57.0), c = swH(i.x + (i.y + 1.0) * 57.0), d = swH(i.x + 1.0 + (i.y + 1.0) * 57.0);
    return 2.0 * mix(mix(a, b, f.x), mix(c, d, f.x), f.y) - 1.0; }
  vec4 swRun(float al, float n, float tau){   /* R, rise time, drain time, time since the front set off */
    float lobes = 1.0 + 0.26 * sin(al * 0.3696 + swN(vec2(al / 60.0, 0.5)) * 1.8) + swN(vec2(al / 6.5, n * 1.37)) * 0.22 + swN(vec2(al / 2.3, n * 2.71 + 5.5)) * 0.12;
    float R = max(uSwash.x * (0.72 + 0.25 * sin(n * 0.9973 + 0.7) + (swH(n + 5000.0) - 0.5) * 0.22) * lobes, 0.05);
    return vec4(R, R * 0.38 + 1.4, R * 0.6 + 2.0, max(tau - swN(vec2(al / 13.0, n * 3.3 + 8.1)) * 0.45, 0.0));
  }
  float swFront(vec4 r){   /* decelerating run-up, then the drain */
    float u = r.w / r.y, v = (r.w - r.y) / r.z;
    return u < 1.0 ? r.x * (1.0 - (1.0 - u) * (1.0 - u)) : max(r.x - (r.x + uSwash.z) * v * v, -uSwash.z);
  }
  float swFilm(vec4 r, float s){   /* covered since the front got here, then a film draining for ~2 s */
    float reach = r.y * (1.0 - sqrt(1.0 - clamp(s / r.x, 0.0, 1.0)));
    float unc = r.y + r.z * sqrt(max((r.x - s) / (r.x + uSwash.z), 0.0));
    return (r.w > reach && s < r.x) ? exp(-max(r.w - unc, 0.0) / 2.2) : 0.0;
  }
  float swStrand(vec4 r, float f, float s){
    return smoothstep(r.y, r.y + 0.3, r.w) * exp(-(r.w - r.y) / 1.7) * (1.0 - smoothstep(0.06, 0.4, abs(s - (r.x - 0.1)))) * smoothstep(0.0, 0.15, s - f);
  }
  vec4 swash(float s, float al, float t, out vec2 sh){
    float n = floor(t / uSwash.y), tau = t - n * uSwash.y;
    vec4 a = swRun(al, n, tau), b = swRun(al, n - 1.0, tau + uSwash.y);
    float fa = swFront(a), fb = swFront(b);
    sh = vec2(min(max(mix(0.035, 0.011, smoothstep(a.y, a.y + 1.8, a.w)) * (fa - s), mix(0.035, 0.011, smoothstep(b.y, b.y + 1.8, b.w)) * (fb - s)), 0.2), max(fa, fb));
    float ua = a.w / a.y;
    float lip = smoothstep(-0.05, 0.05, fa - s) * (1.0 - smoothstep(0.0, ua * 0.8 + 0.45, fa - s)) * (1.0 - smoothstep(0.65, 1.0, ua));
    float sf = smoothstep(0.0, 0.4, fa - s) * exp(-a.w / 2.2) * 0.85;
    return vec4(lip, sf, max(swFilm(a, s), swFilm(b, s) * 0.85), max(swStrand(a, fa, s), swStrand(b, fb, s) * 0.5));
  }`,ne=`
  uniform vec3 uExt, uSeaSh, uSeaNear, uSeaFar; uniform float uSeaGain;
  vec3 seaBody(vec3 p){ return mix(uSeaSh, mix(uSeaNear, uSeaFar, smoothstep(20.0, 110.0, length(p.xz))), smoothstep(0.0, 5.0, -p.y / 0.07)) * uSeaGain; }`;function M(r){return-4.3+(r+6.7)*.281}function T(r,t){const a=L(t)-r;if(a<0)return Math.max(-2.2,a*.07);const o=M(t)-r;let c=Math.min(.35,a*.045)+.05*Math.sin(r*.7+t*.3)*x(a/6,0,1);return o>-1.5&&(c+=.35*x((o+1.5)/4,0,1)**1.5),c}function he(r,t,a,o){const c=z(r,"shore"),n=new q(160,260,160,260);n.rotateX(-A/2),n.translate(-20,0,-90);const v=n.attributes.position,y=[],b=[],D=new f(t.sandDry),d=new f(t.sandPink),S=new f(t.sandWet),C=new f(t.seaShallow),V=new f(t.dune),u=new f,m=!!t.waterExt,l=!!t.swash,W=[];for(let e=0;e<v.count;e++){const s=v.getX(e),p=v.getZ(e);v.setY(e,T(s,p));const h=L(p)-s,k=M(p)-s;W.push(h),h<0?u.copy(S).lerp(C,m?0:x(-h/3,0,1)):h<.8?u.copy(S).lerp(d,h/.8):u.copy(d).lerp(D,x((h-.8)/1.5,0,1)),k>0&&u.lerp(V,x(k/2,0,1)),u.multiplyScalar(t.sandGain??1),y.push(u.r,u.g,u.b),b.push(h<0?1:x(1-h/1.4,0,1)**1.5)}n.setAttribute("color",new $(y,3)),n.computeVertexNormals(),n.setAttribute("wet",new $(b,1)),l&&n.setAttribute("shoreD",new $(W,1));const i=R("Ground090",{repeat:[160/2.2,260/2.2],rough:.95,normal:.9}).clone();i.vertexColors=!0,i.roughnessMap=null,i.roughness=.93,i.normalScale.set(.3,.3);const F=t.causticGain||0,B=F?`${oe}
  float sandCaustic(vec3 p){
    float band = (1.0 - smoothstep(-0.02, -0.003, p.y)) * smoothstep(-0.1, -0.05, p.y) * (1.0 - smoothstep(12.0, 35.0, distance(cameraPosition, p)));
    return band > 0.0 ? band * (caustic(p.xz * 2.2, uTime * 0.16) + 0.6 * caustic(p.xz * 3.7 + 3.0, -uTime * 0.12)) * ${F.toFixed(3)} : 0.0;
  }`:"",O=l?{uSwash:{value:new j(...t.swash)}}:{},X=m?{uExt:{value:new Y(...t.waterExt)},uSeaGain:{value:t.seaGain??.5},uSeaSh:{value:new f(t.seaShallow)},uSeaNear:{value:new f(t.seaNear)},uSeaFar:{value:new f(t.seaFar)}}:{},w=(e,s,p)=>{if(e.split(s).length!==2)throw new Error(`Beach sand: splice anchor not unique: ${s.slice(0,40)}`);return e.replace(s,p)};i.onBeforeCompile=e=>{e.uniforms.uTime=a.uTime,e.uniforms.uSunDir=a.uSunDir;for(const s of Object.keys(o))e.uniforms[s]=o[s];Object.assign(e.uniforms,O,X),e.vertexShader=e.vertexShader.replace("#include <common>",`#include <common>
attribute float wet; varying float vWet; varying vec3 vWp;`).replace("#include <begin_vertex>",`#include <begin_vertex>
vWet = wet; vWp = (modelMatrix * vec4(transformed, 1.0)).xyz;`),e.fragmentShader=e.fragmentShader.replace("#include <common>",`#include <common>
varying float vWet; varying vec3 vWp;
${te}
${ae}${B}`).replace("#include <opaque_fragment>",`outgoingLight *= 1.0 - 0.2 * bcShadow(vWp);${F?`
outgoingLight += diffuseColor.rgb * sandCaustic(vWp);`:""}
#include <opaque_fragment>`).replace("#include <roughnessmap_fragment>",`#include <roughnessmap_fragment>
roughnessFactor = mix(roughnessFactor, 0.78, vWet);`).replace("#include <map_fragment>","vec4 sc = texture2D(map, vMapUv); diffuseColor.rgb *= vec3(dot(sc.rgb, vec3(0.299, 0.587, 0.114)) * 1.5);").replace("#include <color_fragment>",`#include <color_fragment>
diffuseColor.rgb *= mix(1.0, 0.8, vWet);`),l&&(e.vertexShader=w(e.vertexShader,"varying vec3 vWp;","varying vec3 vWp; attribute float shoreD; varying float vShD;"),e.vertexShader=w(e.vertexShader,"vWet = wet;","vWet = wet; vShD = shoreD;"),e.fragmentShader=w(e.fragmentShader,`varying vec3 vWp;
`,`varying vec3 vWp; varying float vShD;
${re}
`),e.fragmentShader=w(e.fragmentShader,"diffuseColor.rgb *= mix(1.0, 0.8, vWet);",`diffuseColor.rgb *= mix(1.0, 0.8, vWet);
vec2 swSh; vec4 swV = swash(vShD, vWp.z, uTime, swSh);
float swCov = smoothstep(-0.03, 0.03, swSh.y - vShD), swFilmV = max(swV.z, swCov) * smoothstep(-0.3, 0.0, vShD);
float swFoamV = clamp((swV.x + swV.y * swCov * 0.5 + swV.w * 0.8) * (0.75 + 0.45 * swN(vWp.xz * vec2(2.2, 1.1))), 0.0, 1.0) * smoothstep(-0.05, 0.0, vShD);
diffuseColor.rgb *= mix(1.0, 0.78, swFilmV);
diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.9), swFoamV * 0.95);`),e.fragmentShader=w(e.fragmentShader,"roughnessFactor = mix(roughnessFactor, 0.78, vWet);",`roughnessFactor = mix(roughnessFactor, 0.78, vWet);
roughnessFactor = mix(mix(roughnessFactor, uSwash.w, swFilmV), 0.9, swFoamV);`)),m&&(e.fragmentShader=w(e.fragmentShader,`#include <common>
`,`#include <common>
${ne}
`),e.fragmentShader=w(e.fragmentShader,`
#include <opaque_fragment>`,`
if (vWp.y < 0.0) {
  float wd = distance(cameraPosition, vWp) * clamp(-vWp.y / max(0.01, cameraPosition.y - vWp.y), 0.0, 1.0);
  ${l?"wd *= 1.0 - smoothstep(0.02, 0.35, vShD - swSh.y);":""}
  vec3 T = exp(-uExt * wd);
  outgoingLight = outgoingLight * T + seaBody(vWp) * (1.0 - T);
}
#include <opaque_fragment>`))},i.customProgramCacheKey=()=>F?`beach-sand-c${F}`:l||m?`beach-sand-w${+l}${+m}`:"beach-sand";const _=new K(n,i);_.receiveShadow=!0,_.userData.dynamic=!0,c.add(_);const I=J(t.footprint,{ramp:"smooth",rim:0,polygonOffset:!0}),E=[];for(let e=0;e<16;e++){const s=-1.2-e*.62,p=.9-e*.12+(e%2?.13:-.13);E.push([p,T(p,s)+.004,s,.25*(e%3),.06,1,.13])}const G=N(new q(1,1).rotateX(-A/2),I,E);G.castShadow=!1;const H=z(c,"footprints");return H.userData.dynamic=!0,H.add(Q(G)),G.userData.dynamic=!0,{g:c,update(){},placeGlitter(){}}}function ve(r,t){const a=z(r,"headlands"),o=Z(21),c=[],n=[],v=(b,D,d,S,C,V)=>{for(let u=0;u<V;u++){const m=b+o()*(D-b),l=d+o()*(S-d),W=1-Math.abs((l-(d+S)/2)/((S-d)/2)),i=.8+o()*1.6+W*C*.4;(o()<.5?c:n).push([l,T(l,m)+i*.3+W*C*.5,m,o()*ee,i*(1+o()*.5),i*.75,i,(o()-.5)*.4])}};v(10,17,-14,14,5,90),v(-71,-75,M(-73)-2,L(-73)+4,1.5,26);const y=new U(1,1);return a.add(N(y,R("Rock051",{repeat:[1,1],tint:16777215,rough:.9,normal:1.4}),c)),a.add(N(y,R("Rock051",{repeat:[1,1],tint:13156568,rough:.9,normal:1.4}),n)),a}export{g as SHORE,re as SWASH_GLSL,ve as buildHeadlands,he as buildShore,T as sandY,L as shoreX,M as vegX};
