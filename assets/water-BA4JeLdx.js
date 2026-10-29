import{I as i,z as l,x as d,e as x,P as g}from"./index-DtL6CxF2.js";import{_ as h,$ as y,L as w}from"./texlocal-LTzE_-Aj.js";const z={ocean:{amp:.16,scale:.12,speed:.35,fres:.62,glit:1,foam:.5,clar:0,oil:0},lake:{amp:.035,scale:.28,speed:.2,fres:.9,glit:.8,foam:0,clar:.2,oil:0},harbor:{amp:.11,scale:.2,speed:.3,fres:.75,glit:.5,foam:0,clar:0,oil:0},canal:{amp:.06,scale:.35,speed:.15,fres:.55,glit:.4,foam:0,clar:0,oil:.1},sludge:{amp:.04,scale:.4,speed:.08,fres:.28,glit:.15,foam:0,clar:0,oil:1}},F=`
  uniform float uWT, uAmp, uScale, uFres, uGlit, uFoam, uOil, uFogNear, uFogFar;
  uniform vec3 uBody, uDeep, uSunCol, uFog;
  uniform vec4 uL[6]; uniform vec3 uLC[6]; uniform int uLN;
  uniform float uHalf;
  uniform vec4 uCalm; /* x, z, radius, strength: a local calm patch (0 strength = off) */
  varying vec3 vWorld; varying vec2 vUv;
  ${y}
  float h21(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
  float vn(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y); }
  vec2 ripple(vec2 p, float t){
    float a = vn(p * vec2(1.0, 2.6) + vec2(t * 0.3, -t * 0.4)), b = vn(p * vec2(1.0, 2.6) + vec2(5.2, 1.3) - vec2(t * 0.2, t * 0.7));
    float c = vn(p * 2.6 + vec2(-t * 0.4, t * 0.6)), d = vn(p * 2.6 + vec2(9.1, 4.7) + vec2(t * 0.3, t * 0.8));
    vec2 s = vec2(0.85 * (a - b), 0.9 * (a - b) + 0.5 * (c - d));
    s += 0.7 * cos(dot(p, vec2(0.16, 0.99)) * 2.1 - t * 0.5) * vec2(0.16, 0.99);
    s += 0.45 * cos(dot(p, vec2(-0.3, 0.95)) * 3.7 - t * 0.6) * vec2(-0.3, 0.95);
    return s;
  }
  void main(){
    vec3 toEye = cameraPosition - vWorld;
    float dist = length(toEye);
    vec3 V = toEye / dist;
    vec2 r = ripple(vWorld.xz * uScale, uWT);
    float near = 1.0 - smoothstep(30.0, 300.0, dist);
    float calm = uCalm.w * (1.0 - smoothstep(uCalm.z * 0.3, uCalm.z, length(vWorld.xz - uCalm.xy)));
    float tilt = uAmp * (0.35 + near) * (1.0 - 0.65 * calm);
    vec3 n = normalize(vec3(r.x * tilt, 1.0, r.y * tilt));
    vec3 R = reflect(-V, n);
    float cosv = clamp(dot(V, n), 0.0, 1.0);
    float F = 0.03 + 0.97 * pow(1.0 - cosv, 5.0);
    vec3 sky = skyColor(normalize(vec3(R.x, max(R.y, 0.015), R.z)));
    sky = mix(sky, sky / (1.0 + sky * 2.4), calm); /* calm patch: highlights (moon glints) soften, darks keep */
    float depthMix = smoothstep(0.0, 1.0, dist / 160.0);
    vec3 body = mix(uBody, uDeep, depthMix * 0.6 + 0.2 * (vn(vWorld.xz * 0.05) - 0.5));
    vec3 col = mix(body, sky * 0.9, clamp(F * uFres + 0.02, 0.0, uFres));
    float s = max(dot(R, uSunDir), 0.0);
    col += uSunCol * (pow(s, 160.0) * 1.1 + pow(s, 40.0) * 0.26 + pow(s, 8.0) * 0.07) * uGlit;
    /* lamp and neon streaks: stretched along the view direction, broken by the ripples */
    vec2 vd = normalize(V.xz + 1e-4);
    for (int i = 0; i < 6; i++) {
      if (i >= uLN) break;
      vec2 off = vWorld.xz - uL[i].xz;
      float along = dot(off, vd), lat = dot(off, vec2(-vd.y, vd.x));
      float w = uL[i].w * (1.0 + 0.15 * r.x);
      float streak = exp(-lat * lat / (w * w)) * exp(-max(along, 0.0) * 0.015) * smoothstep(-2.0, 1.0, along);
      streak *= 0.45 + 0.8 * vn(vec2(lat * 1.2, along * 0.6 + uWT * 0.1));
      col += uLC[i] * streak * (1.0 - cosv * 0.6);
    }
    /* oil sheen (sludge): iridescent bands that drift slowly */
    if (uOil > 0.0) {
      float t = vn(vWorld.xz * 0.35 + uWT * 0.02) + 0.5 * vn(vWorld.xz * 1.1);
      vec3 rb = 0.5 + 0.5 * cos(6.2831 * (t * 1.3 + vec3(0.0, 0.33, 0.67)));
      col += rb * 0.07 * uOil * smoothstep(0.35, 0.8, vn(vWorld.xz * 0.2 + 3.0));
    }
    /* foam streaks on swell crests (ocean) */
    if (uFoam > 0.0) {
      float crest = smoothstep(0.62, 0.9, vn(vWorld.xz * vec2(0.5, 1.1) * uScale * 6.0 + vec2(uWT * 0.1, 0.0)));
      col = mix(col, vec3(1.0), crest * uFoam * 0.22 * (1.0 - smoothstep(30.0, 140.0, dist)));
    }
    col = mix(col, uFog, smoothstep(uFogNear, uFogFar, dist));
    gl_FragColor = vec4(col, 1.0);
  }`;function L(e={}){const a={...z[e.style||"ocean"],...e.params||{}},u=e.size||[400,400],m=e.pos||[0,0,0],p=e.sky||h(e.skyOptions||{}),v=e.lights||[],n=[],f=[];for(let s=0;s<6;s++){const t=v[s];n.push(t?new i(t.pos[0],t.pos[1],t.pos[2],t.size??1.2):new i),f.push(t?new l(t.color):new l(0))}const r=e.fog||{color:10139884,near:80,far:900},c=new d({fog:!1,vertexShader:"varying vec3 vWorld; varying vec2 vUv; void main(){ vUv = uv; vec4 w = modelMatrix * vec4(position, 1.0); vWorld = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }",fragmentShader:F,uniforms:{...p,uWT:{value:0},uAmp:{value:a.amp},uScale:{value:a.scale},uFres:{value:a.fres},uGlit:{value:a.glit},uFoam:{value:a.foam},uOil:{value:a.oil},uFogNear:{value:r.near},uFogFar:{value:r.far},uFog:{value:new l(r.color)},uBody:{value:new l(e.body??812232)},uDeep:{value:new l(e.deep??270432)},uSunCol:{value:new l(e.glitter??16774911)},uL:{value:n},uLC:{value:f},uLN:{value:Math.min(6,v.length)},uHalf:{value:0},uCalm:{value:new i(...e.calm||[0,0,1,0])}}}),o=new x(new g(u[0],u[1],1,1),c);return o.rotation.x=-Math.PI/2,o.position.set(...m),o.name=e.name||"water",o.frustumCulled=!1,o.layers.set(w),o.userData.dynamic=!0,o.userData.update=s=>{c.uniforms.uWT.value=s*a.speed},o.userData.dispose=()=>{o.geometry.dispose(),c.dispose()},o}export{L as w};
