import{z as g}from"./index-DEn5AL3S.js";import{f as d,h as x}from"./texlocal-BzwLwgY3.js";import"./react-three-fiber.esm-BPWxqKlB.js";const i={surf:3.3,floor:-2.5,back:-16,side:9},U={value:0},h=`varying vec3 vUwP;
varying vec3 vUwN;
uniform float uUwT;
uniform float uUwS;
`,y=`
{ float uwh = clamp( position.y, 0.0, 3.0 );
  vec3 uwo = vec3( 0.0 );
  #ifdef USE_INSTANCING
    uwo = instanceMatrix[ 3 ].xyz;
  #endif
  float uwph = uwo.x * 1.7 + uwo.z * 2.3;
  transformed.x += sin( uUwT * 0.8 + uwh * 2.4 + uwph ) * uUwS * uwh * uwh;
  transformed.z += sin( uUwT * 0.65 + uwh * 2.1 + uwph * 1.3 ) * uUwS * 0.6 * uwh * uwh; }
`,S=`
{ vec4 uwp = vec4( transformed, 1.0 );
  vec3 uwn = normal;
  #ifdef USE_INSTANCING
    uwp = instanceMatrix * uwp;
    uwn = mat3( instanceMatrix ) * uwn;
  #endif
  vUwP = ( modelMatrix * uwp ).xyz;
  vUwN = normalize( mat3( modelMatrix ) * uwn ); }
`,P=`
varying vec3 vUwP;
varying vec3 vUwN;
uniform float uUwT;
uniform float uUwK;
uniform float uUwC;
`,$=`
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
  float nearSurf = smoothstep( ${i.floor.toFixed(1)} - 1.0, ${i.surf.toFixed(1)}, P.y );
  gl_FragColor.rgb += vec3( 0.22, 0.68, 0.60 ) * c * up * ( 0.2 + 0.8 * nearSurf ) * uUwC * 0.45;
  /* absorption + in-scatter toward a depth graded water colour */
  gl_FragColor.rgb *= exp( -path * vec3( 0.32, 0.08, 0.065 ) * uUwK );
  float h = smoothstep( -2.2, ${i.surf.toFixed(1)} + 0.3, P.y + 0.12 * cameraPosition.y );
  vec3 deep = vec3( 0.005, 0.024, 0.032 ); /* near-black teal, never #000 at the deep floor and back wall */
  vec3 mid = vec3( 0.003, 0.045, 0.075 );
  vec3 top = vec3( 0.025, 0.28, 0.36 );
  vec3 wc = mix( mix( deep, mid, smoothstep( 0.0, 0.5, h ) ), top, smoothstep( 0.5, 1.0, h ) );
  float fog = 1.0 - exp( -max( 0.0, path - 1.2 ) * 0.15 * uUwK );
  gl_FragColor.rgb = mix( gl_FragColor.rgb, wc, fog ); }
`;function p(e,{k:r=1,caustic:a=1,sway:o=0,surf:c=i.surf,floor:t=i.floor}={}){const l=c!==i.surf||t!==i.floor,s=e.onBeforeCompile,m=e.customProgramCacheKey?e.customProgramCacheKey():"";return e.onBeforeCompile=(n,v)=>{s&&s.call(e,n,v),n.uniforms.uUwT=U,n.uniforms.uUwK={value:r},n.uniforms.uUwC={value:a},n.uniforms.uUwS={value:o};let f=$,w=P;l&&(n.uniforms.uUwSurf={value:c},n.uniforms.uUwFloor={value:t},w+=`uniform float uUwSurf;
uniform float uUwFloor;
`,f=f.replace(`${i.floor.toFixed(1)} - 1.0`,"uUwFloor - 1.0").replace(`${i.surf.toFixed(1)}, P.y`,"uUwSurf, P.y").replace(`${i.surf.toFixed(1)} + 0.3`,"uUwSurf + 0.3")),n.vertexShader=n.vertexShader.replace("#include <common>",`#include <common>
${h}`).replace("#include <project_vertex>",`#include <project_vertex>
${S}`).replace("#include <begin_vertex>",`#include <begin_vertex>
${o>0?y:""}`),n.fragmentShader=n.fragmentShader.replace("#include <common>",`#include <common>
${w}`).replace("#include <fog_fragment>",`${f}
#include <fog_fragment>`)},e.customProgramCacheKey=()=>`uw${o>0?"s":""}${l?`b${c},${t}`:""}|${m}`,e.fog=!1,e.userData.uw=!0,e}const u=new Map;function F(e,r={},a={}){const o=`t|${e}|${JSON.stringify(r,(c,t)=>t&&t.isTexture?t.uuid:t)}|${JSON.stringify(a)}`;return u.has(o)||u.set(o,p(x(e,{...r,uniq:!0}),a)),u.get(o)}function T(e,r={},a={}){const o=`f|${e}|${JSON.stringify(r,(c,t)=>t&&t.isTexture?t.uuid:t)}|${JSON.stringify(a)}`;return u.has(o)||u.set(o,p(d(e,{...r,uniq:!0}),{caustic:0,...a})),u.get(o)}function _(e,r=.25,a={},o={}){const c=new g(e).multiplyScalar(r);return F(e,{...a,emissive:c.getHex()},o)}export{i as W,U as uTime,p as underwater,T as uwFlat,_ as uwLit,F as uwToon};
