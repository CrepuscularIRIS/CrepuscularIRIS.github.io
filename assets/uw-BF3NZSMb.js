import{C as l}from"./index-Dk3Axm_B.js";import{j as w,h as p}from"./texlocal-D77ojHvo.js";import"./react-three-fiber.esm-BT70wUrY.js";const u={surf:3.3,floor:-2.5,back:-16,side:9},m={value:0},g=`varying vec3 vUwP;
varying vec3 vUwN;
uniform float uUwT;
uniform float uUwS;
`,v=`
{ float uwh = clamp( position.y, 0.0, 3.0 );
  vec3 uwo = vec3( 0.0 );
  #ifdef USE_INSTANCING
    uwo = instanceMatrix[ 3 ].xyz;
  #endif
  float uwph = uwo.x * 1.7 + uwo.z * 2.3;
  transformed.x += sin( uUwT * 0.8 + uwh * 2.4 + uwph ) * uUwS * uwh * uwh;
  transformed.z += sin( uUwT * 0.65 + uwh * 2.1 + uwph * 1.3 ) * uUwS * 0.6 * uwh * uwh; }
`,d=`
{ vec4 uwp = vec4( transformed, 1.0 );
  vec3 uwn = normal;
  #ifdef USE_INSTANCING
    uwp = instanceMatrix * uwp;
    uwn = mat3( instanceMatrix ) * uwn;
  #endif
  vUwP = ( modelMatrix * uwp ).xyz;
  vUwN = normalize( mat3( modelMatrix ) * uwn ); }
`,x=`
varying vec3 vUwP;
varying vec3 vUwN;
uniform float uUwT;
uniform float uUwK;
uniform float uUwC;
`,h=`
{ vec3 P = vUwP;
  float full = distance( cameraPosition, P );
  float frac = cameraPosition.z < 0.0 ? 1.0 : clamp( -P.z / max( 1e-3, cameraPosition.z - P.z ), 0.0, 1.0 );
  float path = full * frac;
  /* caustics on up-facing surfaces, fading with depth below the surface */
  /* tiling caustic network (iterated domain warp, bright thin cells), two scales drifting against each other */
  vec2 q = P.xz * 2.0 + vec2( P.y * 0.25, -P.y * 0.15 );
  float c = 0.0;
  for ( int s = 0; s < 2; s ++ ) {
    float sc = s == 0 ? 1.0 : 1.65;
    float tt = uUwT * ( s == 0 ? 0.3 : -0.22 ) + 20.0;
    vec2 pp = mod( q * sc, 6.2831853 ) - 250.0;
    vec2 ii = pp;
    float cc = 1.0;
    for ( int n = 0; n < 4; n ++ ) {
      float t2 = tt * ( 1.0 - 3.5 / float( n + 1 ) );
      ii = pp + vec2( cos( t2 - ii.x ) + sin( t2 + ii.y ), sin( t2 - ii.y ) + cos( t2 + ii.x ) );
      cc += 1.0 / length( vec2( pp.x / ( sin( ii.x + t2 ) / 0.006 ), pp.y / ( cos( ii.y + t2 ) / 0.006 ) ) );
    }
    cc /= 4.0;
    cc = 1.17 - pow( cc, 1.4 );
    c += pow( abs( cc ), 8.0 ) * ( s == 0 ? 1.0 : 0.55 );
  }
  c = clamp( c, 0.0, 1.6 );
  float up = 0.12 + 0.88 * smoothstep( 0.05, 0.8, vUwN.y );
  float nearSurf = smoothstep( ${u.floor.toFixed(1)} - 1.0, ${u.surf.toFixed(1)}, P.y );
  gl_FragColor.rgb += vec3( 0.28, 0.72, 1.0 ) * c * up * ( 0.2 + 0.8 * nearSurf ) * uUwC * 0.55;
  /* absorption + in-scatter toward a depth graded water colour */
  gl_FragColor.rgb *= exp( -path * vec3( 0.3, 0.09, 0.03 ) * uUwK );
  float h = smoothstep( -2.2, ${u.surf.toFixed(1)} + 0.3, P.y + 0.12 * cameraPosition.y );
  vec3 deep = vec3( 0.004, 0.002, 0.045 );
  vec3 mid = vec3( 0.002, 0.03, 0.26 );
  vec3 top = vec3( 0.03, 0.4, 0.85 );
  vec3 wc = mix( mix( deep, mid, smoothstep( 0.0, 0.5, h ) ), top, smoothstep( 0.5, 1.0, h ) );
  float fog = 1.0 - exp( -max( 0.0, path - 1.2 ) * 0.15 * uUwK );
  gl_FragColor.rgb = mix( gl_FragColor.rgb, wc, fog ); }
`;function s(e,{k:i=1,caustic:n=1,sway:t=0}={}){const r=e.onBeforeCompile,o=e.customProgramCacheKey?e.customProgramCacheKey():"";return e.onBeforeCompile=(c,f)=>{r&&r.call(e,c,f),c.uniforms.uUwT=m,c.uniforms.uUwK={value:i},c.uniforms.uUwC={value:n},c.uniforms.uUwS={value:t},c.vertexShader=c.vertexShader.replace("#include <common>",`#include <common>
${g}`).replace("#include <project_vertex>",`#include <project_vertex>
${d}`).replace("#include <begin_vertex>",`#include <begin_vertex>
${t>0?v:""}`),c.fragmentShader=c.fragmentShader.replace("#include <common>",`#include <common>
${x}`).replace("#include <fog_fragment>",`${h}
#include <fog_fragment>`)},e.customProgramCacheKey=()=>`uw${t>0?"s":""}|${o}`,e.fog=!1,e.userData.uw=!0,e}const a=new Map;function U(e,i={},n={}){const t=`t|${e}|${JSON.stringify(i,(r,o)=>o&&o.isTexture?o.uuid:o)}|${JSON.stringify(n)}`;return a.has(t)||a.set(t,s(p(e,{...i,uniq:!0}),n)),a.get(t)}function C(e,i={},n={}){const t=`f|${e}|${JSON.stringify(i,(r,o)=>o&&o.isTexture?o.uuid:o)}|${JSON.stringify(n)}`;return a.has(t)||a.set(t,s(w(e,{...i,uniq:!0}),{caustic:0,...n})),a.get(t)}function $(e,i=.25,n={},t={}){const r=new l(e).multiplyScalar(i);return U(e,{...n,emissive:r.getHex()},t)}export{u as W,m as uTime,s as underwater,C as uwFlat,$ as uwLit,U as uwToon};
