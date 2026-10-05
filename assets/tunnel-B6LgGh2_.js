import{d as t,C as o,b as l}from"./index-DL96uVSh.js";const c=`
  varying vec3 vW;
  void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,i=`
  uniform float uT; uniform vec3 uTop, uMid, uBot;
  varying vec3 vW;
  float h11(float n){ return fract(sin(n * 91.345) * 47453.5453); }
  void main(){
    float u = clamp((vW.y - 0.98) / 0.76, 0.0, 1.0);
    vec3 col = mix(uBot, uMid, smoothstep(0.0, 0.5, u));
    col = mix(col, uTop, smoothstep(0.45, 1.0, u));
    float side = vW.x > 0.0 ? 1.0 : -1.0;
    float zz = vW.z - uT * 0.5 + (side > 0.0 ? 0.0 : 3.1);
    /* concrete ribs every 3 m, slightly lighter than the void between them */
    float rib = smoothstep(0.12, 0.0, abs(fract(zz / 3.0) - 0.5) - 0.38);
    col += vec3(0.006, 0.007, 0.014) * rib;
    /* two cable runs */
    col += vec3(0.05, 0.05, 0.09) * smoothstep(0.012, 0.0, abs(vW.y - 1.12 - 0.015 * sin(zz * 0.4))) * 0.9;
    col += vec3(0.04, 0.04, 0.07) * smoothstep(0.010, 0.0, abs(vW.y - 1.34)) * 0.8;
    /* tunnel lamps: amber caged lamps every 4.6 m with a wide soft bloom, a cooler one half-way */
    float cell = zz / 4.6, f = fract(cell) - 0.5;
    float id = floor(cell);
    float lampHi = 1.52 + 0.06 * (h11(id) - 0.5);
    float dz = f * 4.6, dy = vW.y - lampHi;
    float lamp = exp(-(dz * dz) / 1.1 - (dy * dy) / 0.03);
    float bloom = exp(-(dz * dz) / 2.6 - (dy * dy) / 0.24);
    vec3 amber = vec3(1.0, 0.72, 0.42);
    col += amber * (lamp * 0.5 + bloom * 0.04) * (0.75 + 0.25 * h11(id + 3.0));
    float cell2 = (zz + 2.3) / 4.6, f2 = fract(cell2) - 0.5;
    float d2 = f2 * 4.6, y2 = vW.y - 1.30;
    col += vec3(0.62, 0.70, 1.0) * exp(-(d2 * d2) / 0.25 - (y2 * y2) / 0.012) * 0.28;
    /* now and then a signal: a red or green dot low on the wall */
    float c3 = zz / 31.0, f3 = (fract(c3) - 0.5) * 31.0, id3 = floor(c3);
    vec3 sig = h11(id3 + 9.0) > 0.5 ? vec3(0.2, 1.0, 0.45) : vec3(1.0, 0.18, 0.14);
    col += sig * exp(-(f3 * f3) / 0.12 - pow(vW.y - 1.08, 2.0) / 0.004) * 0.9;
    /* interior reflection: a pale diagonal sheet and the warm red of the benches low in the pane */
    float sheet = smoothstep(0.0, 0.18, 0.5 - abs(u - (0.55 + 0.12 * sin(vW.z * 0.35 + 1.0))));
    col += vec3(0.30, 0.27, 0.46) * sheet * 0.035;
    col += vec3(0.30, 0.06, 0.10) * (1.0 - smoothstep(0.0, 0.35, u)) * 0.05;
    gl_FragColor = vec4(col, 1.0);
  }`;function s(){const e=new t({side:l,fog:!1,vertexShader:c,fragmentShader:i,uniforms:{uT:{value:0},uTop:{value:new o(3816560)},uMid:{value:new o(2369622)},uBot:{value:new o(1579580)}}});return e.userData.update=a=>{e.uniforms.uT.value=a},e}export{s as tunnelGlass};
