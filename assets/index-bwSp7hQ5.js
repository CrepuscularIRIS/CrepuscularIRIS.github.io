import{z as mt,V as Ee,x as gt,G as U,e as k,O as wt,U as He,a5 as Xe,a6 as le,ak as A,$ as he,a0 as Z,P as oe,X as ae,L as J,C as Ne,S as Le,y as Ke,F as E,D as ot,Z as be,d as at,M as ye,a4 as xt,ao as bt,bC as B,bD as G,R as yt,T as Ge,Q as Te,a2 as St,ag as Mt,bE as xe,bF as nt,c as fe,bG as kt,bH as D,bI as zt,as as Pt,bJ as Re,bK as Ce,bL as At,bM as Tt,bN as Oe,bO as Rt,bP as Ct,bQ as Ot,bR as Ve,bS as De}from"./index-BsPPZ0Uo.js";function Dt(){return{uT:{value:0},uD:{value:0},uSpeed:{value:0},uCurve:{value:0},uHill:{value:0},uTunnel:{value:new Ee(1e9,1e9)},uBridge:{value:new Ee(1e9,1e9)},uRamp:{value:1e9},uEnd:{value:1e9},uNight:{value:1},uRain:{value:0},uHead:{value:1},uHeading:{value:0},uFogColor:{value:new mt(1709104)},uFogDensity:{value:.0026}}}const Ye=`
uniform float uT, uD, uSpeed, uCurve, uHill, uRamp, uEnd, uNight, uRain, uHead, uHeading, uFogDensity;
uniform vec2 uTunnel, uBridge;
uniform vec3 uFogColor;
float hash11(float p) { p = fract(p * .1031); p *= p + 33.33; p *= p + p; return fract(p); }
float hash12(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
vec3 bend(vec3 p) { float a = max(-p.z, 0.); p.x += uCurve * a * a; p.y += uHill * a * a; return p; }
// a slot at z0 on a loop of length L that ends 'back' metres behind the car, carried toward the car by the ride
float wrapZ(float z0, float L, float back) { return back - L + mod(z0 + uD - (back - L), L); }
float inTunnel(float s) { return step(uTunnel.x, s) * step(s, uTunnel.y); }
float nearTunnel(float s, float m) { return step(uTunnel.x - m, s) * step(s, uTunnel.y + m); }
float onBridge(float s) { return smoothstep(uBridge.x + 10., uBridge.x + 60., s) * (1. - smoothstep(uBridge.y - 60., uBridge.y, s)); }
float mainStreet(float s) { return smoothstep(uRamp, uRamp + 120., s); }
// colours are written in sRGB terms (as picked); the post chain expects linear light
vec4 lin(vec3 c) { return vec4(pow(max(c, vec3(0.)), vec3(2.2)), 1.); }
float fogOf(float d) { float k = uFogDensity * d; return 1. - exp(-k * k); }
`;function N({uniforms:i,vertexShader:t,fragmentShader:o,...a}){const n=new gt({uniforms:i,vertexShader:Ye+t,fragmentShader:Ye+o,...a});return n.userData.ride=!0,n}const Bt=2;function Qe(i){return i.traverse(t=>{t.raycast=()=>{},t.userData.noCollide=!0,t.userData.dynamic=!0,t.frustumCulled=!1,t.layers.set(Bt)}),i}function Et(i){const t=new U;t.name="ride-sky";const o=new k(new wt(900,32,16),N({uniforms:i,side:Xe,depthWrite:!1,vertexShader:`
        varying vec3 vDir;
        void main() {
          vDir = normalize(position);
          float c = cos(uHeading), s = sin(uHeading);
          vec3 p = vec3(c * position.x - s * position.z, position.y, s * position.x + c * position.z);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.);
          gl_Position.z = gl_Position.w * 0.99999;
        }`,fragmentShader:`
        varying vec3 vDir;
        void main() {
          vec3 d = normalize(vDir);
          float h = clamp(d.y, -0.2, 1.);
          // night: zenith navy, a violet band, a pink haze over the city; dusk: violet zenith over an orange afterglow
          vec3 zen = mix(vec3(.30, .22, .42), vec3(.02, .025, .07), uNight);
          vec3 mid = mix(vec3(.62, .32, .46), vec3(.10, .06, .17), uNight);
          vec3 hor = mix(vec3(1., .55, .32), vec3(.55, .2, .38), uNight);
          vec3 col = mix(hor, mid, smoothstep(0., .16, h));
          col = mix(col, zen, smoothstep(.12, .6, h));
          // stars (night only), twinkling slowly
          vec2 g = vec2(atan(d.z, d.x) * 120., asin(d.y) * 120.);
          vec2 cell = floor(g);
          float st = hash12(cell);
          vec2 f = fract(g) - .5 - (vec2(hash12(cell + 7.), hash12(cell + 3.)) - .5) * .6;
          float star = step(.985, st) * smoothstep(.12, 0., length(f)) * (.6 + .4 * sin(uT * (1. + st * 3.) + st * 40.));
          col += vec3(.85, .9, 1.) * star * smoothstep(.08, .3, h) * uNight * (1. - uRain * .8);
          // the paper moon: ahead-left and high; a soft disc with a folded-paper crease and a halo
          vec3 md = normalize(vec3(-.42, .38, -.82));
          float a = acos(clamp(dot(d, md), -1., 1.));
          float disc = smoothstep(.062, .056, a);
          vec3 tang = normalize(cross(md, vec3(0., 1., 0.)));
          float crease = abs(dot(d - md, tang));
          vec3 paper = vec3(1., .93, .8) * (.92 + .08 * smoothstep(.0, .012, crease)) * (.85 + .15 * hash12(floor(d.xy * 900.)));
          float halo = exp(-a * 9.) * .35 + exp(-a * 30.) * .4;
          float moonOn = mix(.35, 1., uNight) * (1. - uRain * .6);
          col = mix(col, paper, disc * moonOn) + vec3(.9, .75, .7) * halo * moonOn * .6;
          // clouds lit from below by the city: thin bands drifting
          float cl = smoothstep(.55, .9, hash12(floor(vec2(g.x * .02 + uT * .01, g.y * .08)))) * smoothstep(.0, .1, h) * (1. - smoothstep(.25, .5, h));
          col += vec3(.35, .12, .28) * cl * .25 * (1. + uRain);
          gl_FragColor = lin(col);
        }`}));o.renderOrder=-10,t.add(o);const a=new k(new He(650,650,260,96,1,!0),N({uniforms:i,side:Xe,transparent:!0,depthWrite:!1,vertexShader:`
        varying vec2 vUv;
        varying float vY;
        void main() {
          vUv = uv;
          vY = position.y + 130. - 40.;
          float c = cos(uHeading), s = sin(uHeading);
          vec3 p = vec3(c * position.x - s * position.z, position.y + 90., s * position.x + c * position.z);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.);
        }`,fragmentShader:`
        varying vec2 vUv;
        varying float vY;
        void main() {
          float u = vUv.x * 900.;
          float col = floor(u / 3.);
          float wcol = floor(u / 9.);
          // two layers of towers: far (pale) and nearer (darker, taller)
          float h1 = 14. + 42. * pow(hash11(wcol * 1.7), 2.2) + 26. * step(.93, hash11(wcol * 3.1));
          float h2 = 8. + 26. * pow(hash11(floor(u / 6.) * 2.3 + 11.), 2.);
          float y = vY;
          float in1 = step(y, h1);
          float in2 = step(y, h2);
          if (in1 + in2 < .5) {
            // haze above the city glows near the horizon
            float glow = exp(-max(y, 0.) / 35.) * mix(.25, .5, uNight);
            gl_FragColor = vec4(lin(mix(vec3(1., .6, .4), vec3(.75, .3, .55), uNight)).rgb, glow * .55);
            return;
          }
          vec3 body = in2 > .5 ? vec3(.03, .025, .06) : vec3(.07, .05, .11);
          body = mix(body * 2.2 + vec3(.12, .08, .1), body, uNight);
          // windows: a grid, lit at random, warm or cool
          vec2 wg = vec2(u * 1.4, y * .9);
          vec2 cell = floor(wg);
          float lit = step(mix(.9, .72, uNight), hash12(cell + (in2 > .5 ? 50. : 0.)));
          vec2 wf = fract(wg);
          float win = step(.25, wf.x) * step(wf.x, .75) * step(.3, wf.y) * step(wf.y, .7);
          vec3 wc = mix(vec3(1., .78, .5), vec3(.6, .8, 1.), step(.6, hash12(cell * 1.3)));
          vec3 c = body + wc * win * lit * .8 * (in2 > .5 ? .6 : 1.);
          // red beacons blink on the tallest
          float top = in2 > .5 ? h2 : h1;
          float beacon = step(top - 1.2, y) * step(.5, fract(uT * .5 + hash11(wcol))) * step(42., top) * step(abs(fract(u / 9.) - .5), .12);
          c += vec3(1., .1, .08) * beacon * 2.;
          // the haze in front of the far towers
          c = mix(c, mix(vec3(.9, .55, .45), vec3(.42, .2, .38), uNight), .35 * exp(-max(y, 0.) / 60.));
          gl_FragColor = lin(c);
        }`}));return a.renderOrder=-9,t.add(a),t}const ue={mid:-3.6,far:-7.2,oncoming:[-12.2,-15.8,-19.2],parapet:4.6,median:-9.8},Y=32,re=760,ve=14;function Ft(i,t,o=190){const a=new oe(t-i,re,1,o);return a.rotateX(-Math.PI/2),a.translate((i+t)/2,0,ve-re/2),a}function Ht(i){const t=new U;t.name="ride-road";const o=new k(Ft(-22,5.2),N({uniforms:i,vertexShader:`
        varying vec3 vP;
        varying float vS;
        void main() {
          vS = uD - position.z;
          vec3 p = bend(position);
          vP = position;
          vec4 mv = modelViewMatrix * vec4(p, 1.);
          gl_Position = projectionMatrix * mv;
        }`,fragmentShader:`
        varying vec3 vP;
        varying float vS;
        float band(float x, float a, float b) { return step(a, x) * step(x, b); }
        void main() {
          float x = vP.x, s = vS, a = max(-vP.z, 0.);
          float tun = inTunnel(s), bri = onBridge(s);
          // asphalt: dark, fine grain, darker in the wheel tracks
          float grain = hash12(floor(vec2(x * 18., s * 18.))) * .05;
          vec3 col = vec3(.055, .05, .07) + grain;
          col *= 1. - .25 * (exp(-pow((x - .8) * 2.2, 2.)) + exp(-pow((x + .8) * 2.2, 2.)));
          // edge lines (solid) and lane dashes (4.5 m in 12 m)
          float line = band(x, -9.15, -8.98) + band(x, 1.8, 1.95) + band(x, -10.9, -10.75) + band(x, -20.9, -20.75);
          float dash = (band(x, -1.88, -1.72) + band(x, -5.48, -5.32) + band(x, -14.08, -13.92) + band(x, -17.58, -17.42)) * step(fract(s / 12.), .38);
          col = mix(col, vec3(.85, .78, .8), clamp(line + dash, 0., 1.) * .8);
          // the median and the shoulder are rougher concrete
          col *= mix(1., 1.25, band(x, -10.2, -9.4) + band(x, 1.95, 4.4) * .5);
          // light: lamp pools every 32 m (pink-white), the tunnel's sodium bands, our headlights
          float lp = mod(s, ${Y}.), lq = mod(s + 16., ${Y}.);
          float poolR = exp(-pow((lp - 2.) / 9., 2.)) * exp(-pow((x - 1.5) / 6., 2.));
          float poolL = exp(-pow((lq - 2.) / 9., 2.)) * exp(-pow((x + 7.2) / 6.5, 2.));
          vec3 lamp = vec3(1., .72, .8) * (poolR + poolL) * (1. - tun) * .55;
          float sod = exp(-pow((mod(s, 8.) - 4.) / 2.2, 2.)) * tun;
          lamp += vec3(1., .55, .18) * (.25 + .5 * sod) * tun;
          // headlights: a warm fan ahead of the car in our lane
          float fan = smoothstep(2.5, 6., a) * exp(-pow(max(a - 14., 0.) / 26., 2.)) * exp(-pow(x / (1.2 + a * .07), 2.));
          lamp += vec3(1., .93, .8) * fan * uHead * (.55 - .3 * tun);
          col += col * lamp * 6. + lamp * .05;
          // wet: reflections stretched toward the eye (lamp heads, the tail lamps ahead, the neon of the canyon)
          float wet = .35 + .65 * uRain;
          float refR = exp(-pow((x - 2.3) / .35, 2.)) * exp(-pow((lp - 1.) / 14., 2.)) * smoothstep(0., 30., a);
          float refL = exp(-pow((x + 7.2) / .35, 2.)) * exp(-pow((lq - 1.) / 14., 2.)) * smoothstep(0., 30., a);
          vec3 refl = vec3(1., .75, .85) * (refR + refL) * (1. - tun);
          float tail = (exp(-pow((x - .75) / .18, 2.)) + exp(-pow((x + .45) / .18, 2.))) * smoothstep(17., 13., a) * smoothstep(2., 9., a);
          refl += vec3(1., .12, .15) * tail * .9;
          refl += vec3(1., .5, .15) * sod * exp(-pow((x + 3.) / 7., 2.)) * .4;
          // the canyon's neon smeared on the wet road near the barriers
          float neon = (1. - tun) * (1. - bri) * (exp(-pow((x + 8.4) / 1.2, 2.)) + exp(-pow((x - 3.6) / 1.2, 2.))) * .3 * step(.45, hash11(floor(s / 18.))) * smoothstep(4., 40., a);
          float hue = hash11(floor(s / 18.));
          vec3 nc = mix(vec3(1., .2, .75), vec3(.2, .85, 1.), step(.5, hue));
          refl += nc * neon;
          col += refl * wet * .9;
          // puddle sparkle under rain
          col += vec3(.7, .7, .9) * step(.995, hash12(floor(vec2(x * 6., s * 3. - uT * 2.)))) * uRain * .5;
          float d = length(vec3(vP.x, 1.2, vP.z));
          col = mix(col, uFogColor, fogOf(d));
          gl_FragColor = lin(col);
        }`}));o.name="ride-road-deck",t.add(o);const a=new le,n=[],s=[],r=(c,u,m,x)=>{for(let d=0;d<190;d++){const p=ve-re*d/190,w=ve-re*(d+1)/190;n.push(c,u,p,c,m,p,c,u,w,c,m,p,c,m,w,c,u,w);for(let h=0;h<6;h++)s.push(x)}},e=(c,u,m,x)=>{for(let d=0;d<190;d++){const p=ve-re*d/190,w=ve-re*(d+1)/190;n.push(c,m,p,u,m,p,c,m,w,u,m,p,u,m,w,c,m,w);for(let h=0;h<6;h++)s.push(x)}};r(-9.4,0,.85,0),r(-10.2,0,.85,0),e(-10.2,-9.4,.85,1),r(4.4,0,1.15,2),e(4.4,4.9,1.15,3),r(4.9,-1.2,1.15,2),r(-22,-1.2,.9,2),r(5.2,-1.4,0,4),a.setAttribute("position",new A(n,3)),a.setAttribute("kind",new A(s,1));const l=new k(a,N({uniforms:i,side:ae,vertexShader:`
        attribute float kind;
        varying vec3 vP;
        varying float vS, vK;
        void main() {
          vS = uD - position.z; vK = kind; vP = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(bend(position), 1.);
        }`,fragmentShader:`
        varying vec3 vP;
        varying float vS, vK;
        void main() {
          float tun = inTunnel(vS);
          vec3 c = vec3(.16, .13, .18);
          c *= .7 + .3 * hash12(floor(vec2(vS * 2., vP.y * 6.)));
          // pink rim of the lamps along the top
          c += vec3(.5, .25, .4) * smoothstep(.6, .85, vP.y) * (1. - tun) * .5;
          c = mix(c, vec3(.32, .2, .12), tun * .6);
          // joints every 4 m
          c *= .8 + .2 * step(.04, fract(vS / 4.));
          // blue marker lamps on the median every 12 m, and the parapet's lit rail
          float mk = step(abs(vK) , .5) * smoothstep(.25, 0., length(vec2((fract(vS / 12.) - .5) * 12., vP.y - .55))) ;
          c += vec3(.3, .6, 1.) * mk * 3.;
          float rail = step(2.5, vK) * step(vK, 3.5);
          c = mix(c, vec3(1., .75, .85) * (1. - tun * .5), rail * .6);
          c = mix(c, uFogColor, fogOf(length(vP.xz)));
          gl_FragColor = lin(c);
        }`}));return l.name="ride-barriers",t.add(l),t}function Nt(i){const t=[],o=(c,u,m,x,f,d,p)=>{const w=new J(c,u,m);w.translate(x,f,d),w.setAttribute("glow",new A(new Array(w.attributes.position.count).fill(p),1)),w.deleteAttribute("uv"),t.push(w)};o(.22,10,.22,0,5,0,0),o(2.8,.14,.14,-1.4,9.8,0,0),o(.9,.12,.38,-2.6,9.68,0,1);const a=Ue(t),n=Math.ceil(760/Y),s=n*2,r=new he;r.index=a.index;for(const c of Object.keys(a.attributes))r.setAttribute(c,a.attributes[c]);const e=new Float32Array(s*3);for(let c=0;c<n;c++)e.set([ue.parapet+.2,-c*Y+2,1],c*3),e.set([ue.median,-c*Y-14,-1],(n+c)*3);r.setAttribute("slot",new Z(e,3)),r.instanceCount=s;const l=new k(r,N({uniforms:i,vertexShader:`
        attribute float glow;
        attribute vec3 slot;
        varying float vGlow, vS;
        varying vec3 vP;
        void main() {
          vec3 p = position;
          p.x *= slot.z;
          float z = wrapZ(slot.y, ${n*Y}., 14.);
          float s = uD - z;
          float hide = nearTunnel(s, 6.);
          p.y -= hide * 40.;
          p.x += slot.x; p.z += z;
          vGlow = glow; vS = s; vP = p;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(bend(p), 1.);
        }`,fragmentShader:`
        varying float vGlow, vS;
        varying vec3 vP;
        void main() {
          vec3 c = mix(vec3(.08, .07, .1), vec3(1., .85, .9) * 2.2, vGlow);
          c += vec3(.4, .2, .35) * smoothstep(8., 10., vP.y) * (1. - vGlow) * .3;
          c = mix(c, uFogColor, fogOf(length(vP)) * (1. - vGlow * .5));
          gl_FragColor = lin(c);
        }`}));return l.name="ride-lamps",l}function Ue(i){const t=Object.keys(i[0].attributes),o=new le;let a=0;const n=[],s=Object.fromEntries(t.map(r=>[r,[]]));for(const r of i){const e=r;for(const l of t)s[l].push(...e.attributes[l].array);if(e.index)for(const l of e.index.array)n.push(l+a);else for(let l=0;l<e.attributes.position.count;l++)n.push(l+a);a+=e.attributes.position.count}for(const r of t)o.setAttribute(r,new A(s[r],i[0].attributes[r].itemSize));return o.setIndex(n),o}const We=720,it=40,te=-46;function Lt(i){let t=i>>>0;return()=>{t=t+1831565813>>>0;let o=t;return o=Math.imul(o^o>>>15,o|1),o^=o+Math.imul(o^o>>>7,o|61),((o^o>>>14)>>>0)/4294967296}}const Kt=[{face:9.5,side:1,w:[9,18],d:[10,22],h:[22,95],row:0},{face:34,side:1,w:[12,26],d:[14,30],h:[45,170],row:1},{face:-25,side:-1,w:[9,18],d:[10,22],h:[22,95],row:0},{face:-52,side:-1,w:[12,26],d:[14,30],h:[45,170],row:1}];function Ut(i){const t=Lt(20261020),o=[],a=[];for(const n of Kt){if(n.row===1&&!i.midRow)continue;let s=0;for(;s<We-8;){const r=n.w[0]+t()*(n.w[1]-n.w[0]),e=n.d[0]+t()*(n.d[1]-n.d[0]),l=n.h[0]+Math.pow(t(),1.6)*(n.h[1]-n.h[0]),c=1.5+t()*(n.row?10:5),u=n.face+n.side*(e/2+t()*(n.row?14:3)),m=-(s+r/2);if(o.push({x:u,z:m,w:e,d:r,h:l,seed:t(),row:n.row,side:n.side}),n.row===0&&i.signs&&t()<.62){const x=1+(t()<.3?1:0);for(let f=0;f<x;f++){const d=u-n.side*(e/2)-n.side*1.1;a.push({x:d,z:m+r/2-.5-f*(r-1),y:7+t()*14,cell:Math.floor(t()*16),seed:t(),side:n.side})}}s+=r+c}}return{towers:o,signs:a}}function Wt(i,t){const o=new J(1,1,1);o.translate(0,.5,0);const a=new he;a.index=o.index,a.setAttribute("position",o.attributes.position),a.setAttribute("normal",o.attributes.normal);const n=t.towers.length,s=new Float32Array(n*4),r=new Float32Array(n*4);t.towers.forEach((l,c)=>{s.set([l.x,l.z,l.w,l.d],c*4),r.set([l.h,l.seed,l.row,l.side],c*4)}),a.setAttribute("aBox",new Z(s,4)),a.setAttribute("aMeta",new Z(r,4)),a.instanceCount=n;const e=new k(a,N({uniforms:i,vertexShader:`
        attribute vec4 aBox;
        attribute vec4 aMeta;
        varying vec3 vP, vN, vL;
        varying float vSeed, vH, vS, vRow;
        void main() {
          float h = aMeta.x, seed = aMeta.y, row = aMeta.z, side = aMeta.w;
          float zc = wrapZ(aBox.y, ${We}., ${it}.);
          float s = uD - zc;
          float bri = onBridge(s);
          float ms = mainStreet(s);
          // the main street crowds in (nearer, a little lower); on the bridge the towers fall away below the deck
          float x = aBox.x - side * ms * (row < .5 ? 2.5 : 10.);
          float hh = h * (1. - .25 * ms * (1. - row));
          vec3 p = position * vec3(aBox.z, hh - ${te}., aBox.w);
          p.y += ${te}.;
          p.y -= bri * (hh + 120.);
          p.x += x; p.z += zc;
          vL = position * vec3(aBox.z, hh - ${te}., aBox.w) + vec3(0., ${te}., 0.);
          vN = normal; vSeed = seed; vH = hh; vS = s; vRow = row; vP = p;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(bend(p), 1.);
        }`,fragmentShader:`
        varying vec3 vP, vN, vL;
        varying float vSeed, vH, vS, vRow;
        void main() {
          float side = abs(vN.x) > .5 ? 1. : 0.;
          float top = vN.y > .5 ? 1. : 0.;
          // facade coordinate across (u) and up (v)
          float u = side > .5 ? vL.z : vL.x;
          float v = vL.y;
          vec3 base = mix(vec3(.07, .06, .1), vec3(.13, .1, .15), hash11(vSeed * 13.));
          base = mix(base * 1.8 + vec3(.1, .06, .08), base, uNight);
          // floors 3.4 m, bays 1.7 m between piers; windows lit by building, floor and bay, mostly warm
          vec2 cell = floor(vec2(u / 1.7, v / 3.4));
          vec2 f = fract(vec2(u / 1.7, v / 3.4));
          float pane = step(.16, f.x) * step(f.x, .84) * step(.3, f.y) * step(f.y, .78);
          float pier = 1. - step(.08, f.x) * step(f.x, .92);
          float ratio = mix(.16, .52, uNight) * (.6 + .8 * hash11(vSeed * 7.));
          // whole floors lit or dark together, as offices are
          float floorOn = step(.35, hash12(vec2(cell.y, vSeed * 31.)));
          float lit = step(1. - ratio * (.55 + .7 * floorOn), hash12(cell + vSeed * 91.));
          float tint = hash12(cell * 1.7 + vSeed);
          vec3 wc = tint < .62 ? vec3(1., .8, .58) : tint < .86 ? vec3(.7, .84, 1.) : tint < .95 ? vec3(1., .62, .82) : vec3(.62, 1., .86);
          wc *= .55 + .45 * hash12(cell + 3.);
          vec3 c = base * (1. + pier * .35);
          c += wc * pane * lit * step(0., v) * 1.15;
          c += vec3(.05, .06, .09) * pane * (1. - lit) * .5;
          // neon strip lights along the floor slabs of some towers, every few floors
          float every = 3.4 * (3. + floor(hash11(vSeed * 41.) * 4.));
          float dv = abs(fract(v / every + .5) - .5) * every; // metres from the nearest strip
          float strip = step(.6, hash11(vSeed * 37.)) * smoothstep(.3, .08, dv) * step(4., v);
          c += mix(vec3(1., .25, .7), vec3(.25, .85, 1.), step(.5, hash11(vSeed * 43.))) * strip * mix(.5, 1.2, uNight);
          // a big screen on some road-facing walls: slow colour bands and a scan line
          float screen = step(.82, hash11(vSeed * 29.)) * side * step(14., v) * step(v, 14. + min(26., vH * .4)) * step(abs(u), 0.42 * 18.);
          vec3 sc = .5 + .5 * cos(6.28 * (vec3(0., .33, .67) + (v * .03 + uT * .05 + vSeed)));
          sc = mix(sc, vec3(1., .3, .7), .35) * (.75 + .25 * step(.5, fract(v * .8 - uT * 2.)));
          c = mix(c, sc * 1.6, screen * mix(.4, 1., uNight));
          // roof edge lamps on some towers
          float edge = smoothstep(vH - .6, vH - .1, v) * (1. - top) * step(.5, hash11(vSeed * 3.));
          c += mix(vec3(1., .3, .6), vec3(.3, .8, 1.), step(.5, hash11(vSeed * 17.))) * edge * 1.4;
          c = mix(c, base * .8, top);
          // city glow from below, haze with distance
          c += vec3(.22, .08, .18) * exp(-(v - ${te}.) / 40.) * .6;
          float d = length(vP);
          c = mix(c, uFogColor, fogOf(d) * (1. - lit * pane * .35));
          gl_FragColor = lin(c);
        }`}));return e.name="ride-towers",e}const _t=["HOTEL","BAR","24H","KTV","NOODLE","CAFE","ARCADE","NOCTILUCA","JINRUI","GOODBOAT","TAXI","BOOKS","DINER","CLUB","LIVE","OPEN"],Ze=["#ff4fb8","#4fe3ff","#ffb547","#9a7bff","#ff5a5a","#5affc8","#ff7ae0","#7ad0ff"];function It(){const i=document.createElement("canvas");i.width=1024,i.height=1024;const t=i.getContext("2d");t.fillStyle="#000",t.fillRect(0,0,1024,1024),_t.forEach((a,n)=>{const s=n%8*128,r=Math.floor(n/8)*512,e=Ze[n%Ze.length];t.fillStyle="#0b0710",t.fillRect(s+10,r+10,108,492),t.strokeStyle=e,t.lineWidth=5,t.shadowColor=e,t.shadowBlur=16,t.strokeRect(s+16,r+16,96,480),t.fillStyle="#fff",t.font='bold 64px "Oxanium", "Big Shoulders Display", sans-serif',t.textAlign="center",t.textBaseline="middle";const l=a.split(""),c=Math.min(78,440/l.length);l.forEach((u,m)=>{t.shadowBlur=22,t.fillStyle=e,t.fillText(u,s+64,r+256-(l.length-1)*c/2+m*c),t.shadowBlur=0,t.fillStyle="#fff8",t.fillText(u,s+64,r+256-(l.length-1)*c/2+m*c)})});const o=new Ne(i);return o.colorSpace=Le,o.anisotropy=4,o}function $t(i,t){if(!t.signs.length)return null;const o=new oe(1.7,6.8),a=new he;a.index=o.index,a.setAttribute("position",o.attributes.position),a.setAttribute("uv",o.attributes.uv);const n=t.signs.length,s=new Float32Array(n*4),r=new Float32Array(n*2);t.signs.forEach((c,u)=>{s.set([c.x,c.z,c.y,c.cell],u*4),r.set([c.seed,c.side],u*2)}),a.setAttribute("aSign",new Z(s,4)),a.setAttribute("aMeta",new Z(r,2)),a.instanceCount=n;const e=It(),l=new k(a,N({uniforms:{...i,uAtlas:{value:e}},transparent:!0,depthWrite:!1,blending:Ke,side:ae,vertexShader:`
        attribute vec4 aSign;
        attribute vec2 aMeta;
        varying vec2 vUv;
        varying float vFl, vFog;
        void main() {
          float zc = wrapZ(aSign.y, ${We}., ${it}.);
          float s = uD - zc;
          float hide = onBridge(s) + inTunnel(s);
          float ms = mainStreet(s);
          vec3 p = position;
          p.x += aSign.x - aMeta.y * ms * 2.5;
          p.y += aSign.z - hide * 400.;
          p.z += zc;
          float cell = aSign.w;
          // canvas rows run top-down; the texture's v runs bottom-up (flipY)
          vUv = vec2((mod(cell, 8.) + uv.x) / 8., 1. - (floor(cell / 8.) + 1. - uv.y) / 2.);
          // neon flicker: mostly steady, now and then a stutter
          float k = floor(uT * 12. + aMeta.x * 100.);
          vFl = .82 + .18 * sin(uT * 3. + aMeta.x * 40.);
          vFl *= step(.04, hash11(k * .37 + aMeta.x)) ;
          vFog = fogOf(length(p));
          gl_Position = projectionMatrix * modelViewMatrix * vec4(bend(p), 1.);
        }`,fragmentShader:`
        uniform sampler2D uAtlas;
        varying vec2 vUv;
        varying float vFl, vFog;
        void main() {
          vec3 c = texture2D(uAtlas, vUv).rgb;
          c *= vFl * mix(.55, 1.3, uNight) * (1. - vFog);
          gl_FragColor = vec4(c, 1.);
        }`}));return l.name="ride-signs",l.userData.textures=[e],l}function qt(i){const t=new oe(2400,2400,1,1);t.rotateX(-Math.PI/2),t.translate(0,te+6,-800);const o=new k(t,N({uniforms:i,vertexShader:`
        varying vec3 vP;
        void main() { vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.); }`,fragmentShader:`
        varying vec3 vP;
        void main() {
          float s = uD - vP.z;
          float x = vP.x;
          float bri = onBridge(s);
          // lower city: dark blocks, glowing streets every 60 m across and 45 m along, moving car lights on them
          vec2 q = vec2(x, s);
          vec2 st = abs(fract(q / vec2(45., 60.)) - .5) * vec2(45., 60.);
          float street = exp(-pow(min(st.x, st.y) / 1.6, 2.));
          vec3 city = vec3(.02, .015, .03) + vec3(1., .55, .3) * street * .35;
          float car = step(.93, hash12(floor(vec2(x / 45., (s + uT * 14.) / 3.)))) * exp(-pow(st.x / .6, 2.));
          city += vec3(1., .9, .7) * car * .8;
          // the bay: dark water, slow swell, plankton glowing in drifting patches, the far shore's lights reflected
          float w1 = sin(x * .05 + s * .02 + uT * .4) + sin(x * .013 - s * .041 - uT * .3) + sin(x * .031 + s * .057 + uT * .2);
          // smooth value noise for the plankton patches (no blocks)
          vec2 nq = q / 26.;
          vec2 ni = floor(nq), nf = fract(nq);
          nf = nf * nf * (3. - 2. * nf);
          float n = mix(mix(hash12(ni), hash12(ni + vec2(1., 0.)), nf.x), mix(hash12(ni + vec2(0., 1.)), hash12(ni + vec2(1., 1.)), nf.x), nf.y);
          float patchy = smoothstep(.45, 1.5, w1 * .45 + n * 1.2);
          float sparkle = step(.985, hash12(floor(q / 1.5) + floor(uT * 2.)));
          vec3 sea = vec3(.01, .03, .05) + vec3(.1, .95, .85) * patchy * .22 + vec3(.5, 1., .95) * sparkle * patchy * .8;
          float refl = pow(abs(sin(x * .021 + 1.3)), 40.) * .3;
          sea += vec3(.9, .5, .7) * refl * exp(-abs(s - uD) / 900.);
          vec3 c = mix(city, sea, bri);
          float d = length(vP - vec3(0., ${te}., 0.));
          c = mix(c, uFogColor, fogOf(d) * .85);
          gl_FragColor = lin(c);
        }`}));return o.name="ride-ground",o.renderOrder=-8,o}const I=(i,t,o,a,n)=>i.push(...t,...o,...a,...t,...a,...n);function st(i,t,o={}){return N({uniforms:i,side:ae,...o,vertexShader:`
      attribute float aS;
      attribute float aK;
      varying vec3 vP;
      varying float vS, vK;
      void main() {
        vec3 p = position;
        p.z += -(aS - uD);
        // far behind or beyond the drawn road: folded away
        if (p.z > 60. || p.z < -900.) p.y -= 2000.;
        vP = p; vS = aS; vK = aK;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(bend(p), 1.);
      }`,fragmentShader:t})}function jt(i,[t,o]){const a=[],n=[],s=[];for(let x=t;x<o;x+=4){const f=a.length;I(a,[-10.3,0,0],[-10.3,6.4,0],[-10.3,6.4,-4],[-10.3,0,-4]),I(a,[5,0,0],[5,0,-4],[5,6.4,-4],[5,6.4,0]),I(a,[-10.3,6.4,0],[5,6.4,0],[5,6.4,-4],[-10.3,6.4,-4]);for(let d=f;d<a.length;d+=3)n.push(x),s.push(d-f<36?0:1);if(Math.round((x-t)/4)%2===0)for(const d of[-10.3+.25,5-.25]){const p=a.length,w=6.4-.9,h=6.4-.6,v=-.4,S=-2.8,M=d+(d<0?.18:-.18);I(a,[M,w,v],[M,h,v],[M,h,S],[M,w,S]),I(a,[d,w,v],[M,w,v],[M,w,S],[d,w,S]);for(let z=p;z<a.length;z+=3)n.push(x),s.push(2)}}for(const[x,f]of[[t,1],[o,-1]]){const d=a.length,p=0;I(a,[-34,0,p],[-10.3,0,p],[-10.3,24,p],[-34,24,p]),I(a,[5,0,p],[30,0,p],[30,24,p],[5,24,p]),I(a,[-10.3,6.4,p],[5,6.4,p],[5,24,p],[-10.3,24,p]);for(let w=d;w<a.length;w+=3)n.push(x),s.push(3);if(f>0){const w=a.length;I(a,[-6,6.4+1.2,p+.05],[2.5,6.4+1.2,p+.05],[2.5,6.4+3,p+.05],[-6,6.4+3,p+.05]);for(let h=w;h<a.length;h+=3)n.push(x),s.push(4)}}const u=new le;u.setAttribute("position",new A(a,3)),u.setAttribute("aS",new A(n,1)),u.setAttribute("aK",new A(s,1));const m=new k(u,st(i,`
    varying vec3 vP;
    varying float vS, vK;
    void main() {
      float s = uD - vP.z;
      vec3 c;
      if (vK < .5) {
        // tiled walls: cream tiles, a dark dado, grime toward the road
        vec2 t = fract(vec2(s / .6, vP.y / .3));
        float joint = step(.06, t.x) * step(.08, t.y);
        c = mix(vec3(.32, .26, .2), vec3(.62, .52, .38), step(1.4, vP.y)) * (.75 + .25 * joint);
        c *= .55 + .45 * smoothstep(0., 2., vP.y);
        // sodium light falling from the lamps every 8 m
        c *= .45 + .9 * exp(-pow((mod(s, 8.) - 4.6) / 2.6, 2.)) * smoothstep(0., 5.5, vP.y + 1.);
        c *= vec3(1., .78, .5);
      } else if (vK < 1.5) {
        c = vec3(.09, .07, .06) * (.6 + .4 * exp(-pow((mod(s, 8.) - 4.6) / 2., 2.)));
      } else if (vK < 2.5) {
        c = vec3(1., .72, .35) * 3.;
      } else if (vK < 3.5) {
        c = vec3(.2, .18, .22) * (.8 + .2 * hash12(floor(vP.xy * 2.)));
        c += vec3(1., .6, .3) * .25 * exp(-abs(vP.y - 6.6) * 1.5) * step(abs(vP.x + 2.6), 8.);
      } else {
        // the name board over the entrance: white letters drawn as bars
        vec2 q = vec2((vP.x + 6.) / 8.5, (vP.y - 7.6) / 1.8);
        float letters = step(.3, fract(q.x * 9.)) * step(.25, q.y) * step(q.y, .75) * step(.08, q.x) * step(q.x, .92);
        c = mix(vec3(.05, .25, .2), vec3(.95), letters);
      }
      c = mix(c, uFogColor * vec3(1., .8, .6), fogOf(length(vP)) * .8);
      gl_FragColor = lin(c);
    }`));return m.name="ride-tunnel",m}function Xt(i,[t,o]){const a=[],n=(f,d,p,w,h,v,S,M)=>{const z=new J(d-f,w-p,v-h);z.translate((f+d)/2,(p+w)/2,(h+v)/2),z.deleteAttribute("uv"),z.deleteAttribute("normal");const L=z.attributes.position.count;z.setAttribute("aS",new A(new Array(L).fill(S),1)),z.setAttribute("aK",new A(new Array(L).fill(M),1)),a.push(z)},s=t+170,r=o-190,e=92,l=-24,c=8;for(const f of[s,r]){for(const d of[l,c])n(d-1.6,d+1.6,-46,e,-1.6,1.6,f,0),n(d-.5,d+.5,e,e+1.2,-.5,.5,f,2);for(const d of[28,82])n(l,c,d,d+3,-1.2,1.2,f,0)}const u=f=>{if(f<s)return 2+(e-2)*Math.max(0,(f-t)/(s-t))**1.6;if(f>r)return 2+(e-2)*Math.max(0,(o-f)/(o-r))**1.6;const d=(f-s)/(r-s);return e-(e-10)*4*d*(1-d)};for(const f of[l+1,c-1]){for(let d=t;d<o;d+=6){const p=u(d),w=u(d+6),h=new le,v=[];I(v,[f-.35,p,0],[f+.35,p,0],[f+.35,w,-6],[f-.35,w,-6]),I(v,[f,p-.35,0],[f,p+.35,0],[f,w+.35,-6],[f,w-.35,-6]),h.setAttribute("position",new A(v,3)),h.setAttribute("aS",new A(new Array(12).fill(d),1)),h.setAttribute("aK",new A(new Array(12).fill(1),1)),a.push(h)}for(let d=t+12;d<o-6;d+=12){const p=u(d);p<3||n(f-.06,f+.06,1,p,-.06,.06,d,1)}}const m=Ue(a.map(f=>f.index?f.toNonIndexed():f)),x=new k(m,st(i,`
    varying vec3 vP;
    varying float vS, vK;
    void main() {
      vec3 c;
      if (vK < .5) {
        // towers: pale concrete washed by uplights from the deck, darker toward the top
        // towers: floodlit from the deck, pale and warm, with a row of lamps along each crossbeam
        c = vec3(.78, .72, .8) * (.45 + .55 * exp(-max(vP.y, 0.) / 70.));
        c += vec3(1., .75, .85) * .3 * exp(-abs(vP.y - 2.) / 14.);
        c += vec3(1., .9, .75) * step(.82, fract(vP.x * .5)) * (step(abs(vP.y - 29.5), .25) + step(abs(vP.y - 83.5), .25)) * 2.;
      } else if (vK < 1.5) {
        // cables and hangers: lit by small lamps along them
        c = vec3(.75, .8, 1.) * (.35 + .65 * step(.8, fract(vP.y * .2 + vP.z * .05)));
      } else {
        c = vec3(1., .1, .08) * 4. * step(.5, fract(uT * .6));
      }
      c = mix(c, uFogColor, fogOf(length(vP)) * (vK > 1.5 ? .3 : 1.));
      gl_FragColor = lin(c);
    }`));return x.name="ride-bridge",x}const Gt=[["BAY TUNNEL","HARBOUR  2 km"],["BAY BRIDGE","NOCTILUCA BAY"],["EXIT 1 km","MAIN STREET"],["MAIN STREET","KERB  ·  CENTRE"]];function Vt(){const i=document.createElement("canvas");i.width=1024,i.height=512;const t=i.getContext("2d");Gt.forEach(([a,n],s)=>{const r=s%2*512,e=Math.floor(s/2)*256;t.fillStyle="#0c5a4a",t.fillRect(r,e,512,256),t.strokeStyle="#e8f4ee",t.lineWidth=8,t.strokeRect(r+14,e+14,484,228),t.fillStyle="#f4faf6",t.textAlign="center",t.font='bold 64px "Oxanium", "Big Shoulders Display", sans-serif',t.fillText(a,r+256,e+110),t.font='44px "Oxanium", "Big Shoulders Display", sans-serif',t.fillText(n,r+256,e+186)});const o=new Ne(i);return o.colorSpace=Le,o.anisotropy=4,o}function Yt(i,t){const o=[],a=[],n=[],s=[],r=(m,x,f,d)=>{o.push(...m),a.push(...x),n.push(f),s.push(d)},e=(m,x,f,d,p,w)=>{for(const h of[p,p-.5]){const v=[[m,f,h],[x,f,h],[x,d,h],[m,f,h],[x,d,h],[m,d,h]];for(const S of v)r(S,[0,0],w,0)}};t.forEach(({s:m,board:x})=>{e(-10,-9.6,0,8.2,0,m),e(4.6,5,0,8.2,0,m),e(-10,5,7.6,8.2,0,m);const f=x%2/2,d=1-(Math.floor(x/2)+1)/2,p=-6.8,w=.6,h=4.8,v=7.5,S=.06,M=[[p,h,S,f,d],[w,h,S,f+.5,d],[w,v,S,f+.5,d+.5],[p,h,S,f,d],[w,v,S,f+.5,d+.5],[p,v,S,f,d+.5]];for(const[z,L,K,W,C]of M)r([z,L,K],[W,C],m,1)});const l=new le;l.setAttribute("position",new A(o,3)),l.setAttribute("uv",new A(a,2)),l.setAttribute("aS",new A(n,1)),l.setAttribute("aK",new A(s,1));const c=Vt(),u=new k(l,N({uniforms:{...i,uBoards:{value:c}},side:ae,vertexShader:`
      attribute float aS;
      attribute float aK;
      varying vec3 vP;
      varying vec2 vUv;
      varying float vK;
      void main() {
        vec3 p = position;
        p.z += -(aS - uD);
        if (p.z > 60. || p.z < -900.) p.y -= 2000.;
        vP = p; vUv = uv; vK = aK;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(bend(p), 1.);
      }`,fragmentShader:`
      uniform sampler2D uBoards;
      varying vec3 vP;
      varying vec2 vUv;
      varying float vK;
      void main() {
        vec3 c = vK > .5 ? texture2D(uBoards, vUv).rgb * (.55 + .45 * uHead) * 1.1 : vec3(.12, .11, .14);
        c = mix(c, uFogColor, fogOf(length(vP)));
        gl_FragColor = lin(c);
      }`}));return u.name="ride-gantries",u.userData.textures=[c],u}const rt=520,Qt=40;function Zt(){const i=[],t=(o,a,n,s,r,e)=>{const l=new J(o,a,n);l.translate(0,s,r),l.deleteAttribute("uv"),l.setAttribute("kind",new A(new Array(l.attributes.position.count).fill(e),1)),i.push(l)};return t(1.82,.62,4.4,.62,0,0),t(1.5,.5,2.2,1.18,.25,1),Ue(i)}function Jt(){const i=[],t=[],o=[],a=(r,e,l,c,u,m)=>{const x=[[r,l,u],[e,l,u],[e,c,u],[r,l,u],[e,c,u],[r,c,u]],f=[[0,0],[1,0],[1,1],[0,0],[1,1],[0,1]];x.forEach((d,p)=>{i.push(...d),t.push(...f[p]),o.push(m)})},n=(r,e,l,c,u,m)=>{const x=[[r,l,c],[e,l,c],[e,l,u],[r,l,c],[e,l,u],[r,l,u]],f=[[0,0],[1,0],[1,1],[0,0],[1,1],[0,1]];x.forEach((d,p)=>{i.push(...d),t.push(...f[p]),o.push(m)})};a(-1.1,1.1,.35,.95,-2.22,0),a(-1.1,1.1,.5,1,2.22,1);for(const r of[-.7,.7]){const e=[[r,.55,-2.2],[r,.85,-2.2],[r,.85,18],[r,.55,-2.2],[r,.85,18],[r,.55,18]],l=[[0,0],[0,1],[1,1],[0,0],[1,1],[1,0]];e.forEach((c,u)=>{i.push(...c),t.push(...l[u]),o.push(2)})}n(-1.6,1.6,.03,-2.2,-14,3);const s=new le;return s.setAttribute("position",new A(i,3)),s.setAttribute("uv",new A(t,2)),s.setAttribute("kind",new A(o,1)),s}function eo(i){const t=[];let o=7;const a=()=>(o=o*16807%2147483647)/2147483647,n=Math.round(i*.4);for(let s=0;s<i;s++){const r=s>=n,e=r?ue.oncoming:[ue.mid,ue.far];t.push({x:e[Math.floor(a()*e.length)]+(a()-.5)*.4,z0:-a()*rt,v:r?18+a()*12:16+a()*10,dir:r?1:-1,paint:a()})}return t}const Je=`
  attribute vec4 aCar; // x, z0, speed, dir (+1 oncoming, -1 our way)
  attribute float aPaint;
  attribute float kind;
  varying vec3 vP;
  varying float vKind, vPaint, vRel, vDir, vS;
  varying vec2 vUv;
  void main() {
    float dir = aCar.w;
    // oncoming cars come toward us at their own speed plus ours; our way, they move ahead of us at theirs
    float z = wrapZ(aCar.y + dir * aCar.z * uT, ${rt}., ${Qt}.);
    vec3 p = position;
    if (dir > 0.) { p.x = -p.x; p.z = -p.z; }
    float s = uD - z;
    p.x += aCar.x; p.z += z;
    vRel = dir > 0. ? aCar.z + uSpeed : abs(uSpeed - aCar.z);
    vKind = kind; vPaint = aPaint; vDir = dir; vS = s; vP = p;
    #ifdef USE_UV
    vUv = uv;
    #endif
    gl_Position = projectionMatrix * modelViewMatrix * vec4(bend(p), 1.);
  }`;function et(i,t){const o=new he;i.index&&(o.index=i.index);for(const s of Object.keys(i.attributes))o.setAttribute(s,i.attributes[s]);const a=new Float32Array(t.length*4),n=new Float32Array(t.length);return t.forEach((s,r)=>{a.set([s.x,s.z0,s.v,s.dir],r*4),n[r]=s.paint}),o.setAttribute("aCar",new Z(a,4)),o.setAttribute("aPaint",new Z(n,1)),o.instanceCount=t.length,o}function to(i,t,{trails:o=!0}={}){const a=new U;a.name="ride-traffic";const n=new k(et(Zt(),t),N({uniforms:i,vertexShader:Je,fragmentShader:`
        varying vec3 vP;
        varying float vKind, vPaint, vRel, vDir, vS;
        void main() {
          vec3 paint = vPaint < .3 ? vec3(.09, .09, .11) : vPaint < .55 ? vec3(.5, .52, .58) : vPaint < .8 ? vec3(.24, .06, .1) : vec3(.05, .1, .2);
          float lamp = exp(-pow((mod(vS, 32.) - 2.) / 8., 2.)) * (1. - inTunnel(vS));
          vec3 c = paint * (.25 + .9 * lamp) + vec3(.08, .05, .09);
          c = mix(c, vec3(.03, .03, .05) + vec3(.4, .2, .35) * lamp * .3, step(.5, vKind));
          c *= mix(vec3(1.), vec3(1.2, .85, .55), inTunnel(vS));
          c = mix(c, uFogColor, fogOf(length(vP)));
          gl_FragColor = lin(c);
        }`}));n.name="ride-traffic-bodies",a.add(n);const s=Jt(),r=new k(et(s,t),N({uniforms:{...i,uTrails:{value:o?1:0}},transparent:!0,depthWrite:!1,blending:Ke,side:ae,defines:{USE_UV:""},vertexShader:Je,fragmentShader:`
        uniform float uTrails;
        varying vec3 vP;
        varying float vKind, vPaint, vRel, vDir, vS;
        varying vec2 vUv;
        void main() {
          vec3 c = vec3(0.);
          float fog = 1. - fogOf(length(vP)) * .9;
          if (vKind < .5) {
            // headlamps: two hot dots with a soft halo (seen from the front, i.e. oncoming)
            vec2 q = vec2(abs(vUv.x - .5) * 2.2 - .62, (vUv.y - .5) * .6);
            float d = length(q * vec2(1., 2.2));
            c = vec3(.92, .95, 1.) * (exp(-d * d * 120.) * 3. + exp(-d * 9.) * .35);
          } else if (vKind < 1.5) {
            vec2 q = vec2(abs(vUv.x - .5) * 2.2 - .7, (vUv.y - .5) * .5);
            float d = length(q * vec2(1., 2.4));
            c = vec3(1., .1, .12) * (exp(-d * d * 160.) * 2.6 + exp(-d * 10.) * .3);
          } else if (vKind < 2.5) {
            // trails: fade along their length and with relative speed; white-blue toward us, red away
            float a = (1. - vUv.x) * smoothstep(0., .08, vUv.x) * exp(-pow((vUv.y - .5) * 3.2, 2.));
            vec3 tc = vDir > 0. ? vec3(.55, .7, 1.) : vec3(1., .12, .18);
            c = tc * a * smoothstep(4., 30., vRel) * .55 * uTrails;
          } else {
            // the oncoming cars' headlamp pools on the road
            float a = exp(-pow((vUv.x - .5) * 2.6, 2.)) * (1. - vUv.y) * smoothstep(0., .1, vUv.y);
            c = vec3(.6, .65, .75) * a * .12 * step(0., vDir);
          }
          gl_FragColor = lin(c * fog);
        }`}));return r.name="ride-traffic-lamps",r.renderOrder=3,a.add(r),a}const pe=new E(0,1.067,-1.142),Fe=new E(0,1.325,-.565),oo=1.78,lt=[-.78,.02],ct=1.75;function ao(i,t){if(!t)return null;const o=new oe(.018,.9),a=new he;a.index=o.index,a.setAttribute("position",o.attributes.position),a.setAttribute("uv",o.attributes.uv);const n=new Float32Array(t*4);let s=3;const r=()=>(s=s*16807%2147483647)/2147483647;for(let l=0;l<t;l++)n.set([(r()-.5)*30,r()*18,-r()*70,r()],l*4);a.setAttribute("aDrop",new Z(n,4)),a.instanceCount=t;const e=new k(a,N({uniforms:i,transparent:!0,depthWrite:!1,blending:Ke,vertexShader:`
        attribute vec4 aDrop;
        varying float vA;
        varying vec2 vUv;
        void main() {
          // fall at 9 m/s, drift toward the car with the ride; loop in a box around the cabin
          float y = mod(aDrop.y - uT * 9. * (1. + aDrop.w * .3), 18.) - 2.;
          float z = mod(aDrop.z + uD + uT * 2., 70.) - 64.;
          vec3 p = position;
          // lean with the wind of the ride
          p.z += p.y * min(uSpeed * .04, .9);
          p += vec3(aDrop.x, y, z);
          // never inside the cabin
          if (abs(p.x) < 1.1 && p.z > -2.6 && p.z < 2.6 && p.y < 1.6) p.y += 40.;
          vA = uRain * (.25 + .5 * aDrop.w) * (1. - inTunnel(uD - z));
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.);
        }`,fragmentShader:`
        varying float vA;
        varying vec2 vUv;
        void main() {
          float a = vA * smoothstep(0., .3, vUv.y) * (1. - abs(vUv.x - .5) * 2.);
          gl_FragColor = lin(vec3(.75, .8, .95) * a * 1.1);
        }`}));return e.name="ride-rain",e}function no(i,t){const o=pe.distanceTo(Fe),a=new oe(oo,o,1,1),n=new k(a,N({uniforms:{...i,uWipe:t.uniforms.uWipe,uWipeRate:t.uniforms.uWipeRate,uPivots:{value:new Ee(...lt)}},transparent:!0,depthWrite:!1,vertexShader:`
        varying vec2 vQ;
        void main() {
          vQ = vec2(position.x, position.y + ${(o/2).toFixed(4)});
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.);
        }`,fragmentShader:`
        uniform float uWipe, uWipeRate;
        uniform vec2 uPivots;
        varying vec2 vQ;
        // how long ago (s) a blade last swept this point: a blade sweeps 0..SWEEP and back once per 1/uWipeRate s
        float age(vec2 q, float px) {
          vec2 d = q - vec2(px, .02);
          float r = length(d);
          if (r > .78 || uWipeRate <= 0.) return 99.;
          float ang = atan(d.y, d.x); // 0 = resting along the base (the blade lies toward +x), up to SWEEP
          float a = clamp(ang / ${ct.toFixed(3)}, 0., 1.);
          float ph = fract(uWipe); // uWipe counts sweeps (up and back), uWipeRate sweeps per second
          float now = ph < .5 ? ph * 2. : 2. - ph * 2.; // blade position 0..1..0
          // time since the blade passed a: within this period, either on the way up or on the way down
          float up = a * .5, down = 1. - a * .5;
          float since = ph >= down ? ph - down : ph >= up ? ph - up : ph + (1. - down);
          return since / uWipeRate;
        }
        void main() {
          vec2 q = vQ;
          float ag = min(age(q, uPivots.x), age(q, uPivots.y));
          float grow = clamp(ag / 2.4, 0., 1.);
          // drops: a jittered grid at two scales; each drop is a lens: a faint body, a dark lower rim and a small
          // bright highlight up and to the left (the city lights it), not a ring
          float body = 0., spec = 0.;
          for (int k = 0; k < 2; k++) {
            float sc = k == 0 ? 26. : 47.;
            vec2 g = q * sc;
            vec2 c = floor(g);
            vec2 f = fract(g) - .5;
            float h = hash12(c + float(k) * 13.);
            vec2 o = (vec2(hash12(c + 5.), hash12(c + 9.)) - .5) * .5;
            float born = fract(h * 7. + uT * .05 * (1. + h));
            float on = step(1. - uRain * (k == 0 ? .32 : .45) * grow, h);
            float r = (k == 0 ? .2 : .15) * (.55 + .45 * hash12(c + 2.));
            vec2 d = f - o - vec2(0., -born * .1);
            float disc = smoothstep(r, r * .8, length(d));
            body += on * disc * (.35 + .65 * smoothstep(-r, r, -d.y));
            spec += on * smoothstep(r * .45, 0., length(d - vec2(-r * .3, r * .35)));
          }
          // the beads catch the city's colours (pink from the left, cyan from the right)
          vec3 tint = mix(vec3(1., .6, .85), vec3(.6, .9, 1.), smoothstep(-.8, .8, q.x));
          float a = clamp(body * .38 + spec * 1.2, 0., 1.) * uRain;
          gl_FragColor = vec4(lin(mix(tint * .35, tint, clamp(spec, 0., 1.))).rgb, a);
        }`})),s=Fe.clone().sub(pe).normalize(),r=new E(1,0,0).cross(s).normalize();return n.position.copy(pe).addScaledVector(s,o/2).addScaledVector(r,-.008),n.quaternion.setFromRotationMatrix(new ot().makeBasis(new E(1,0,0),s,r)),n.name="ride-windscreen-drops",n.renderOrder=5,n}function io(){const i=Fe.clone().sub(pe).normalize(),t=new E(1,0,0).cross(i).normalize(),o=new ot().makeBasis(new E(1,0,0),i,t),a=new U;a.name="ride-wipers";const n=[],s=new be({color:723984,roughness:.6,metalness:.3});for(const r of lt){const e=new U;e.position.copy(pe).addScaledVector(t,-.025).add(new E(r,0,0)).addScaledVector(i,.02),e.quaternion.setFromRotationMatrix(o);const l=new U;e.add(l);const c=new k(new J(.72,.016,.014),s);c.position.set(.36,0,0);const u=new k(new J(.66,.026,.02),s);u.position.set(.4,0,-.008),l.add(c,u),a.add(e),n.push(l)}return{group:a,blades:n,material:s}}const T={x:.36,z0:-.36,z1:.38,y:1.33};function so(i){i.updateMatrixWorld(!0);const t=new at,o=new E,a=new E;i.traverse(n=>{if(!n.isMesh||n.name)return;const s=n.material;s!=null&&s.isMeshBasicMaterial&&s.transparent&&(t.setFromObject(n).getSize(o),t.getCenter(a),o.x<.05&&o.z>2.4&&o.z<2.8&&Math.abs(a.x)>.7&&a.y>1&&a.y<1.4&&(n.name=a.x>0?"glass-right":"glass-left"))})}function ro(i,t,o){let a=null,n=1/0;const s=new E;return i.traverse(r=>{if(r.name!==t)return;r.getWorldPosition(s);const e=o?s.distanceTo(o):0;e<n&&(n=e,a=r)}),a}function lo(i,t){const o=[],a=new Map,n=new at;return i.updateMatrixWorld(!0),i.traverse(s=>{if(!s.isMesh||s.userData.ride||s.userData.noCollide||[].concat(s.material).some(e=>e==null?void 0:e.isShaderMaterial)||(n.setFromObject(s),n.max.y<T.y||n.min.x>T.x||n.max.x<-.36||n.min.z>T.z1||n.max.z<T.z0))return;const r=e=>{if(!e||e.transparent)return e;if(!a.has(e)){const l=e.clone();l.onBeforeCompile=c=>{c.uniforms.uRoofOpen=t,c.vertexShader=c.vertexShader.replace("void main() {",`varying vec3 vRoofW;
void main() {`).replace("#include <project_vertex>",`#include <project_vertex>
  vRoofW = (modelMatrix * vec4(transformed, 1.0)).xyz;`),c.fragmentShader=c.fragmentShader.replace("void main() {",`uniform float uRoofOpen;
varying vec3 vRoofW;
void main() {
  if (uRoofOpen > 0.001 && vRoofW.y > ${T.y.toFixed(3)} && abs(vRoofW.x) < ${T.x.toFixed(3)} && vRoofW.z > ${T.z0.toFixed(3)} && vRoofW.z < ${T.z0.toFixed(3)} + ${(T.z1-T.z0).toFixed(3)} * uRoofOpen) discard;`)},l.customProgramCacheKey=()=>"ride-roof",a.set(e,l),o.push(l)}return a.get(e)};s.material=Array.isArray(s.material)?s.material.map(r):r(s.material)}),o}function co(){const i=document.createElement("canvas");i.width=512,i.height=224;const t=new Ne(i);t.colorSpace=Le;const o=new k(new oe(.31,.13),new ye({map:t,toneMapped:!1}));o.position.set(-.012,.983,-.9115),o.rotation.x=-.08,o.name="ride-screen";let a="";return{mesh:o,draw:s=>{const r=JSON.stringify(s);if(r===a)return;a=r;const e=i.getContext("2d");e.fillStyle="#050b14",e.fillRect(0,0,512,224),e.fillStyle="rgba(120,170,230,0.5)",e.fillRect(12,11,488,1),e.font="11px sans-serif",e.fillStyle="#7fa8d8",e.fillText(s.clock,15,25),e.fillText("MEDIA   ROUTE   CAR",190,25);const l=s.station>0;if(e.fillStyle=l?"#13243a":"#0b1420",e.fillRect(14,38,200,172),e.fillStyle=l?"#ff9ab8":"#4a6a8a",e.font="bold 15px sans-serif",e.fillText(xe[s.station].name,24,62),e.font="12px sans-serif",e.fillStyle="#9cc4ee",l){e.fillText(xe[s.station].track,24,84),e.fillText(xe[s.station].artist,24,102);for(let c=0;c<14;c++){const u=8+(c*37+s.beat*13)%40;e.fillStyle="#4aa8ff",e.fillRect(24+c*13,190-u,8,u)}}else e.fillText("F at the screen: tune in",24,84);e.strokeStyle="#2a4a6a",e.lineWidth=4,e.beginPath(),e.moveTo(250,120),e.lineTo(490,120),e.stroke(),e.strokeStyle="#4aa8ff",e.beginPath(),e.moveTo(250,120),e.lineTo(250+240*s.progress,120),e.stroke(),nt.forEach((c,u)=>{const m=250+240*c.t/G;e.fillStyle=s.chapter>=c.n?"#ff9ab8":"#2a4a6a",e.beginPath(),e.arc(m,120,6,0,Math.PI*2),e.fill(),e.fillStyle="#7fa8d8",e.font="10px sans-serif",e.fillText(String(u+1),m-3,142)}),e.fillStyle="#ffffff",e.beginPath(),e.arc(250+240*s.progress,120,8,0,Math.PI*2),e.fill(),e.fillStyle="#9cc4ee",e.font="bold 13px sans-serif",e.fillText(s.title,250,80),e.font="12px sans-serif",e.fillText(s.eta,250,176),t.needsUpdate=!0},tex:t}}function fo(){const i=new U;i.name="ride-charm";const t=new k(new He(.0012,.0012,.11,4),new ye({color:10127984}));t.position.y=-.055;const o=new St;o.absarc(0,0,.032,Math.PI*.15,Math.PI*1.85,!1),o.absarc(.014,0,.026,Math.PI*1.75,Math.PI*.25,!0);const a=new k(new Mt(o,{depth:.003,bevelEnabled:!1,curveSegments:18}),new be({color:15918792,emissive:6969920,emissiveIntensity:.45,roughness:.9,side:ae}));return a.position.set(0,-.14,-.0015),a.rotation.z=.25,i.add(t,a),i.position.set(0,1.262,-.63),i}function vo(){const i=new U;i.name="ride-lights";const t=new yt(4868704,2102296,.34),o=new Ge(16758984,1.3);o.position.set(20,30,-40),o.target.position.set(0,0,-10),o.castShadow=!0,o.shadow.mapSize.set(2048,2048),Object.assign(o.shadow.camera,{left:-30,right:30,top:30,bottom:-30,near:1,far:120}),o.shadow.bias=-4e-4,o.shadow.normalBias=.03;const a=new Te(16761032,60,16,1.4);a.position.set(1.5,7,-20);const n=new Te(16730832,30,9,1.6);n.position.set(-7,2,-12);const s=new Ge(8030412,.9);s.position.set(-.2,1.7,1),s.target.position.set(.1,.85,-1);const r=new Te(4881151,.2,.8,2);return r.position.set(-.2,.95,-.75),i.add(t,o,o.target,a,n,s,s.target,r),{group:i,hemi:t,key:o,pool:a,mag:n,fill:s,disp:r}}function uo(i,t){const o=new U;o.name="ride-cabin";const a={value:0},n=lo(i,a),s=i.getObjectByName("sunroof");s&&(s.visible=!1);const r=new be({color:11845078,roughness:.32,metalness:.35,emissive:1316900}),e=new U;e.name="ride-sunroof";const l=T.x*2,c=T.z1-T.z0,u=(T.z0+T.z1)/2;for(const[b,y,ie,V]of[[l+.06,.03,0,T.z0-.015],[l+.06,.03,0,T.z1+.015],[.03,c,-.36-.015,u],[.03,c,T.x+.015,u]]){const ge=new k(new J(b,.012,y),r);ge.position.set(ie,1.392,V),e.add(ge)}const m=new ye({color:3816026,transparent:!0,opacity:.35,depthWrite:!1,side:ae}),x=new k(new oe(l,c),m);x.rotation.x=Math.PI/2,x.position.set(0,1.455,u),o.add(e,x);const f=fo();o.add(f);const d=co();o.add(d.mesh);const p=new ye({color:3866522,toneMapped:!1}),w=new be({color:2764344,roughness:.45,metalness:.5}),h=new U;h.name="ride-lightswitch",h.position.set(-.72,.79,-.858);const v=new k(new He(.024,.026,.022,20).rotateX(Math.PI/2),w);v.position.z=.011;const S=new k(new J(.004,.018,.004),p);S.position.set(0,.01,.023),v.add(S);const M=new k(new xt(.005,12),p);M.position.set(.04,.012,.002),h.add(v,M),o.add(h),n.push(r,m,p,w,d.mesh.material,d.tex);for(const b of[f,e,x,d.mesh,h])b.traverse(y=>{y.userData.ride=!0,y.raycast=()=>{}});o.traverse(b=>{b.userData.noCollide=!0});const z=i.getObjectByName("wheel"),L=z==null?void 0:z.quaternion.clone(),K=ro(i,"glass-right",new E(1,1.2,0)),W=(K==null?void 0:K.position.y)??0,C=i.getObjectByName("leadcar"),ce=C==null?void 0:C.position.clone(),Se=new E(0,0,1),me=new bt,ne={t0:-99,amp:0};return{group:o,owned:n,uniforms:{uRoofOpen:a},flick(b){ne.t0=b,ne.amp=.75},screen:d,update(b,y){a.value=b.roof,x.position.z=u+c*b.roof*.98,x.visible=b.roof>.01,K&&(K.position.y=W-.44*b.window),z&&z.quaternion.copy(L).multiply(me.setFromAxisAngle(Se,B.clamp(-b.curve*900,-.5,.5)));const ie=y-ne.t0,V=ne.amp*Math.exp(-ie/2.2)*Math.sin(ie*5.2);f.rotation.x=V+b.accel*.03+Math.sin(y*1.7)*.015*Math.min(1,b.speed/10),f.rotation.z=B.clamp(b.curve*b.speed*b.speed*.5,-.4,.4)+V*.2,p.color.setHex(b.head?3866522:1058330),v.rotation.z=b.head?-.9:0,C&&(C.position.x=ce.x-1+Math.sin(y*.31)*.12,C.position.z=ce.z-Math.max(0,b.t-(G-14))**2*.9,C.visible=C.position.z>-400)},dispose(){var b;for(const y of n)(b=y.dispose)==null||b.call(y)}}}const tt={high:{traffic:46,rain:1400,signs:!0,midRow:!0,trails:!0},medium:{traffic:34,rain:800,signs:!0,midRow:!0,trails:!0},low:{traffic:24,rain:360,signs:!0,midRow:!1,trails:!1},software:{traffic:14,rain:0,signs:!1,midRow:!1,trails:!1}},we={from:"kouka",title:"Kouka",text:"Saw your car come out of the bay tunnel on the traffic feed. The water is glowing again tonight, the plankton bloom. Look left before the bridge ends."},Be=i=>{try{return new URLSearchParams(location.search).get(i)}catch{return null}};function po(i){var o,a;const t=i.getObjectByName("leadcar");t&&t.parent!==i&&i.attach(t);for(const n of["road","wet","streaks","lights","skyDome"]){const s=i.getObjectByName(n);s&&(s.visible=!1)}i.traverse(n=>{var s;(s=n.userData)!=null&&s.update&&/mist/i.test(n.name||"")&&n.parent===i&&(n.visible=!1)});for(const n of i.children)n.isMesh&&((a=(o=n.material)==null?void 0:o.uniforms)!=null&&a.uSpeed)&&n!==t&&(n.visible=!1)}function mo(i,{quality:t}={}){const o=fe.tier||(t==="low"?"low":"high"),a=tt[o]||tt.high,n=Be("tod")?Be("tod")!=="day":kt(),s=Be("rain")!=="0",r=i.root;so(r),po(r),i.walk={seated:!0,bounds:[-.3,-.3,.3,.3],eye:1.2,floorY:()=>0,colliders:[],start:{position:[0,1.2,.02],target:[0,1.06,-1]},fov:64},i.env=i.env||{},i.env.fog=n?{color:1709104,density:.0026}:{color:5913688,density:.0024},i.env.background=n?395278:3811908;const e=Dt();e.uTunnel.value.set(...D.tunnel),e.uBridge.value.set(...D.bridge),e.uRamp.value=D.ramp,e.uEnd.value=D.end,e.uNight.value=n?1:0,e.uFogColor.value.setHex(i.env.fog.color,zt),e.uFogDensity.value=i.env.fog.density;const l=new U;l.name="ride";const c=Ut({midRow:a.midRow,signs:a.signs}),u=[],m=g=>{g&&(l.add(g),u.push(...g.userData.textures||[]))};m(Et(e)),m(qt(e)),m(Ht(e)),m(Nt(e)),m(Wt(e,c)),m($t(e,c)),m(jt(e,D.tunnel)),m(Xt(e,D.bridge)),m(Yt(e,[{s:420,board:0},{s:D.tunnel[0]-170,board:0},{s:D.bridge[0]+90,board:1},{s:D.bridge[1]-260,board:2},{s:D.ramp+60,board:3}])),m(to(e,eo(a.traffic),{trails:a.trails})),m(ao(e,s?a.rain:0)),Qe(l),r.add(l);const x={uniforms:{uWipe:{value:0},uWipeRate:{value:0}}},f=uo(r),d=Qe(no(e,x)),p=io();p.group.traverse(g=>{g.raycast=()=>{},g.userData.noCollide=!0});const w=vo();r.add(f.group,d,p.group,w.group);const h=At(),v={t:0,chapter:1,station:0,roof:0,window:0,wipers:0,head:!0,arrived:!1,rain:0},S={roof:0,window:0},M=new Set,z=new Set,L=g=>{for(const R of z)try{R(g)}catch(F){console.error(F)}};let K=performance.now(),W=0,C=!1;const ce=()=>{K=performance.now()};document.addEventListener("visibilitychange",ce);const Se=(g,R={})=>L({kind:"toast",text:g,...R}),me={"radio:next":()=>{h.arm();const g=h.setStation(h.stationIndex+1===xe.length?0:h.stationIndex+1);return v.station=h.stationIndex,L({kind:"radio",station:v.station}),g.url?`${g.name} · "${g.track}", ${g.artist}`:"The radio clicks off. Just the road now."},"radio:off":()=>(h.arm(),h.setStation(0),v.station=0,L({kind:"radio",station:0}),"The radio clicks off. Just the road now."),"sunroof:toggle":()=>(h.arm(),h.click(),S.roof=S.roof>.5?0:1,S.roof?"The sunroof slides back. Cold air, and the towers leaning in overhead.":"The sunroof slides shut. The city sound drops away."),"window:toggle":()=>(h.arm(),h.click(),S.window=S.window>.5?0:1,S.window?"The window winds down. The wind comes in loud, smelling of rain and the sea.":"The window winds up. Quiet again."),"wipers:cycle":()=>(h.arm(),h.click("wiper"),v.wipers=(v.wipers+1)%3,v.wipers===0&&(C=!0),["Wipers off.","Wipers on.","Wipers fast."][v.wipers]+(v.rain<.05&&v.wipers?" The glass is dry; they squeak once and keep time.":"")),"lights:toggle":()=>(h.arm(),h.click(v.head?"off":"switch"),v.head=!v.head,v.head?"Headlamps on. The lane comes back out of the dark.":"Headlamps off. Between the street lamps the road goes dark; the car does not mind."),"charm:flick":()=>(f.flick(ne()),"You flick the paper moon. It spins on its thread, and the dashboard light runs around its edge."),"route:show":()=>{const g=Ve(v.t),R=Math.max(0,Math.round(G-v.t));return v.arrived?"Main street, at the kerb. The route screen says: you have arrived.":`Chapter ${g.n} of 3, ${g.title}. Main street in ${Math.floor(R/60)}:${String(R%60).padStart(2,"0")}.`},"ride:restart":()=>(b.seek(0),M.clear(),"The car pulls away from the kerb again."),horn:()=>(h.arm(),h.click(),null)},ne=()=>performance.now()/1e3,b={get state(){return{...v,length:Ce,end:G}},message:we,subscribe(g){return z.add(g),()=>z.delete(g)},use(g){var H,O,q,P;const R=(q=(O=(H=g==null?void 0:g.interactable)==null?void 0:H.primary)==null?void 0:O.effect)==null?void 0:q.device;if(!R||!me[R])return!1;const $=me[R]()||((P=g.lines)==null?void 0:P[0])||null;return $&&Se($,{device:R}),!0},seek(g){v.t=Math.max(0,Math.min(Ce,g)),K=performance.now(),g<Re.mid&&M.delete("mid"),g<G&&(M.delete("arrive"),v.arrived=!1)},audio:h};i.ride=b;const y={pos:new E,rx:0,ry:0,rz:0,on:!1,p0:new E,r0:new Pt},ie=(g,R,F,$)=>{const H=performance.now(),O=Math.min(1,(H-K)/1e3);K=H;const q=document.hidden||!!fe.hold||!!fe.settling;q||(v.t=Math.min(Ce,v.t+O));const P=v.t,_=Tt(P),j=Oe(P),ee=Rt(P),ke=(Oe(P+.25)-Oe(P-.25))/.5;e.uT.value=H/1e3%3600,e.uD.value=_,e.uSpeed.value=j,e.uCurve.value=ee,e.uHill.value=Ct(P),e.uHeading.value=Ot(P),v.rain=s?B.clamp((P-Re.rain)/10,0,1)*(P>G+2?.8:1):0,e.uRain.value=v.rain,e.uHead.value=B.lerp(e.uHead.value,v.head?1:0,Math.min(1,O*6)),v.roof+=B.clamp(S.roof-v.roof,-O*.7,O*.7),v.window+=B.clamp(S.window-v.window,-O*.6,O*.6);const ze=[0,.75,1.35][v.wipers];if(ze)W+=O*ze,C=!1;else if(C){const Q=Math.ceil(W);W=Math.min(Q,W+O*.75),W>=Q&&(C=!1)}x.uniforms.uWipe.value=W,x.uniforms.uWipeRate.value=ze||(C?.75:0);const Pe=W%1,ft=Pe<.5?Pe*2:2-Pe*2;for(const Q of p.blades)Q.rotation.z=ft*ct;f.update({t:P,roof:v.roof,window:v.window,curve:ee,speed:j,accel:ke,head:v.head},H/1e3);const se=Ve(P),_e=Math.max(0,Math.round(G-P));f.screen.draw({station:v.station,beat:v.station?Math.floor(H/500):0,chapter:se.n,title:v.arrived?"Arrived: Main street":`${se.n} · ${se.title}`,progress:Math.min(1,P/G),eta:v.arrived?"At the kerb":`Main street ${Math.floor(_e/60)}:${String(_e%60).padStart(2,"0")}`,clock:new Date().toTimeString().slice(0,5)});const X=_>D.tunnel[0]&&_<D.tunnel[1]?1:0,de=_>D.bridge[0]+40&&_<D.bridge[1]-40?1:0,vt=_%Y/Y,Ie=Math.exp(-(((vt-.5)*Y/7)**2)),$e=Math.exp(-(((_%8-4)/2.2)**2)),{key:qe,hemi:Ae,pool:ut,mag:je,fill:pt}=w;qe.intensity=X?.6+.9*$e:(de?.95:1.1)+.35*Ie,qe.color.setHex(X?16754768:de?12110079:16758984),Ae.intensity=X?.2+.35*$e:de?.42:.3+.22*Ie,Ae.color.setHex(X?16752736:4868704),Ae.groundColor.setHex(de?932410:X?3810320:2102296),ut.intensity=(X?25:60)*(v.head?1:.35);const ht=Math.floor(_/18)%3;if(je.color.setHex(X?16748608:de?4247776:[16730832,4905215,16742970][ht]),je.intensity=X?10:30,pt.intensity=n?.9:1.15,h.update({speed:j,wind:Math.max(v.window,v.roof*.6),rain:v.rain,tunnel:X,paused:q}),$!=null&&$.warp&&ge($.warp,{v:j,accel:ke,tun:X,D:_,camera:F}),(se.n!==v.chapter||!M.has("chapter1"))&&(v.chapter=se.n,M.add("chapter1"),L({kind:"chapter",chapter:se})),!M.has("start")&&P>2.5&&(M.add("start"),De("ride:start",{place:"car-1",chapters:nt.length,length:G})),!M.has("mid")&&P>=Re.mid&&(M.add("mid"),De("ride:mid",{place:"car-1",where:"bay-bridge",hie:we.from,message:we.text}),L({kind:"message",message:we})),!M.has("arrive")&&P>=G&&(M.add("arrive"),v.arrived=!0,De("ride:arrive",{place:"car-1",to:"street-1"}),L({kind:"arrive"})),F&&!fe.still){const Q=Math.min(1,j/20);y.pos.set(Math.sin(H*.0021)*.002*Q,(Math.sin(H*.0083)*.0022+Math.sin(H*.0191)*.0011)*Q+(dt(H)-.5)*6e-4*Q,0),y.rz=B.clamp(-ee*j*j*.035,-.035,.035),y.ry=B.clamp(ee*18,-.03,.03),y.rx=B.clamp(-ke*.004,-.02,.02),y.p0.copy(F.position),y.r0.copy(F.rotation),F.position.add(y.pos),F.rotation.x+=y.rx,F.rotation.y+=y.ry,F.rotation.z+=y.rz,y.on=!0}},V=new E,ge=(g,{v:R,accel:F,tun:$,D:H,camera:O})=>{const q=fe.still||o==="software"||!O,P=B.smoothstep(R,17,25),_=B.clamp(F/3,0,1),j=H-D.tunnel[1];if(g.amount=q?0:Math.min(.85,.5*P+.3*_+.25*$),g.lines=q?0:Math.min(.6,B.smoothstep(F,.3,1)*P+.35*$),g.flash=q||j<0?0:.8*Math.exp(-j/10),g.near=2.5,!q){const ee=B.smoothstep(-O.getWorldDirection(V).z,.35,.75);g.amount*=ee,g.lines*=ee,V.set(0,1.2,-100).project(O),g.center[0]=B.clamp(V.x*.5+.5,0,1),g.center[1]=B.clamp(V.y*.5+.5,0,1)}},dt=g=>{const R=Math.sin(g*12.9898)*43758.5453;return R-Math.floor(R)};i.update=ie,i.afterRender=g=>{y.on&&(g.position.copy(y.p0),g.rotation.copy(y.r0),y.on=!1)};const Me=i.dispose;return i.dispose=()=>{Me==null||Me(),document.removeEventListener("visibilitychange",ce),h.dispose(),f.dispose(),z.clear(),l.traverse(g=>{g.isMesh&&(g.geometry.dispose(),g.material.dispose())});for(const g of u)g.dispose();d.geometry.dispose(),d.material.dispose(),p.material.dispose()},b}export{mo as attachRide};
