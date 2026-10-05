import{g as f,C as l,d,M as x,P as g}from"./index-DSoSip_m.js";import{J as y,S as h,L as w}from"./texlocal-OZM9ruvh.js";const F={ocean:{amp:.16,scale:.12,speed:.35,fres:.62,glit:1,foam:.5,clar:0,oil:0},lake:{amp:.035,scale:.28,speed:.2,fres:.9,glit:.8,foam:0,clar:.2,oil:0},harbor:{amp:.11,scale:.2,speed:.3,fres:.75,glit:.5,foam:0,clar:0,oil:0},canal:{amp:.06,scale:.35,speed:.15,fres:.55,glit:.4,foam:0,clar:0,oil:.1},sludge:{amp:.04,scale:.4,speed:.08,fres:.28,glit:.15,foam:0,clar:0,oil:1}},z=`
  uniform float uWT, uAmp, uScale, uFres, uGlit, uFoam, uOil, uFogNear, uFogFar;
  uniform vec3 uBody, uDeep, uSunCol, uFog;
  uniform vec4 uL[6]; uniform vec3 uLC[6]; uniform int uLN;
  uniform float uHalf;
  varying vec3 vWorld; varying vec2 vUv;
  ${h}
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
    vec3 n = normalize(vec3(r.x * uAmp * (0.35 + near), 1.0, r.y * uAmp * (0.35 + near)));
    vec3 R = reflect(-V, n);
    float cosv = clamp(dot(V, n), 0.0, 1.0);
    float F = 0.03 + 0.97 * pow(1.0 - cosv, 5.0);
    vec3 sky = skyColor(normalize(vec3(R.x, max(R.y, 0.015), R.z)));
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
  }`;function S(e={}){const a={...F[e.style||"ocean"],...e.params||{}},v=e.size||[400,400],p=e.pos||[0,0,0],m=e.sky||y(e.skyOptions||{}),i=e.lights||[],u=[],n=[];for(let s=0;s<6;s++){const t=i[s];u.push(t?new f(t.pos[0],t.pos[1],t.pos[2],t.size??1.2):new f),n.push(t?new l(t.color):new l(0))}const r=e.fog||{color:10139884,near:80,far:900},c=new d({fog:!1,vertexShader:"varying vec3 vWorld; varying vec2 vUv; void main(){ vUv = uv; vec4 w = modelMatrix * vec4(position, 1.0); vWorld = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }",fragmentShader:z,uniforms:{...m,uWT:{value:0},uAmp:{value:a.amp},uScale:{value:a.scale},uFres:{value:a.fres},uGlit:{value:a.glit},uFoam:{value:a.foam},uOil:{value:a.oil},uFogNear:{value:r.near},uFogFar:{value:r.far},uFog:{value:new l(r.color)},uBody:{value:new l(e.body??812232)},uDeep:{value:new l(e.deep??270432)},uSunCol:{value:new l(e.glitter??16774911)},uL:{value:u},uLC:{value:n},uLN:{value:Math.min(6,i.length)},uHalf:{value:0}}}),o=new x(new g(v[0],v[1],1,1),c);return o.rotation.x=-Math.PI/2,o.position.set(...p),o.name=e.name||"water",o.frustumCulled=!1,o.layers.set(w),o.userData.dynamic=!0,o.userData.update=s=>{c.uniforms.uWT.value=s*a.speed},o.userData.dispose=()=>{o.geometry.dispose(),c.dispose()},o}export{S as w};
