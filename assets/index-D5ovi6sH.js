import{z as xt,V as We,x as bt,G as K,e as S,O as yt,U as Ie,a0 as Qe,Z as de,$ as P,a5 as me,a6 as ee,P as oe,X as ie,L as te,C as ze,S as Pe,y as $e,F as B,D as it,a4 as Me,d as st,M as pe,ac as St,bC as $,ao as kt,bD as D,R as Mt,T as Be,Q as Fe,a2 as zt,ak as Pt,bE as ke,bF as rt,c as le,bG as Tt,bH as O,bI as At,as as Rt,bJ as Ne,bK as He,bL as Et,bM as Dt,bN as Le,bO as Ct,bP as Ot,bQ as Bt,bR as Ze,bS as Ue}from"./index-IY9d3W7M.js";import{paintCluster as Ft}from"./ctex-sKTUpjy3.js";import"./texlocal-CpJ0jTLc.js";import"./react-three-fiber.esm-BIVGIr2o.js";function Nt(){return{uT:{value:0},uD:{value:0},uSpeed:{value:0},uCurve:{value:0},uHill:{value:0},uTunnel:{value:new We(1e9,1e9)},uBridge:{value:new We(1e9,1e9)},uRamp:{value:1e9},uEnd:{value:1e9},uNight:{value:1},uRain:{value:0},uHead:{value:1},uHeading:{value:0},uFogColor:{value:new xt(1709104)},uFogDensity:{value:.0026}}}const Je=`
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
`;function L({uniforms:i,vertexShader:t,fragmentShader:a,...o}){const n=new bt({uniforms:i,vertexShader:Je+t,fragmentShader:Je+a,...o});return n.userData.ride=!0,n}const Ht=2;function et(i){return i.traverse(t=>{t.raycast=()=>{},t.userData.noCollide=!0,t.userData.dynamic=!0,t.frustumCulled=!1,t.layers.set(Ht)}),i}function Lt(i){const t=new K;t.name="ride-sky";const a=new S(new yt(900,32,16),L({uniforms:i,side:Qe,depthWrite:!1,vertexShader:`
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
        }`}));a.renderOrder=-10,t.add(a);const o=new S(new Ie(650,650,260,96,1,!0),L({uniforms:i,side:Qe,transparent:!0,depthWrite:!1,vertexShader:`
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
        }`}));return o.renderOrder=-9,t.add(o),t}const he={mid:-3.6,far:-7.2,oncoming:[-12.2,-15.8,-19.2],parapet:4.6,median:-9.8},Y=32,ce=760,ue=14;function Ut(i,t,a=190){const o=new oe(t-i,ce,1,a);return o.rotateX(-Math.PI/2),o.translate((i+t)/2,0,ue-ce/2),o}function Kt(i){const t=new K;t.name="ride-road";const a=new S(Ut(-22,5.2),L({uniforms:i,vertexShader:`
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
          float paint = clamp(line + dash, 0., 1.);
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
          // the paint is lit like the asphalt but by a gain that saturates: under a lamp or the headlights a line is a
          // pale stripe, not a white bloom (a straight lit gain on near-white paint blew out to a pink-white halo)
          float gain = 1. + 6. * dot(lamp, vec3(.3, .5, .2));
          col = mix(col, vec3(.8, .75, .76) * (.3 + .45 * gain / (1. + .45 * gain)) + lamp * .03, paint * .85);
          // wet: reflections stretched toward the eye (lamp heads, the tail lamps ahead, the neon of the canyon)
          float wet = .22 + .78 * uRain; // dry road a faint sheen, the rain takes it to full
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
        }`}));a.name="ride-road-deck",t.add(a);const o=new de,n=[],s=[],r=(c,v,p,x)=>{for(let d=0;d<190;d++){const h=ue-ce*d/190,w=ue-ce*(d+1)/190;n.push(c,v,h,c,p,h,c,v,w,c,p,h,c,p,w,c,v,w);for(let m=0;m<6;m++)s.push(x)}},e=(c,v,p,x)=>{for(let d=0;d<190;d++){const h=ue-ce*d/190,w=ue-ce*(d+1)/190;n.push(c,p,h,v,p,h,c,p,w,v,p,h,v,p,w,c,p,w);for(let m=0;m<6;m++)s.push(x)}};r(-9.4,0,.85,0),r(-10.2,0,.85,0),e(-10.2,-9.4,.85,1),r(4.4,0,1.15,2),e(4.4,4.9,1.15,3),r(4.9,-1.2,1.15,2),r(-22,-1.2,.9,2),r(5.2,-1.4,0,4),o.setAttribute("position",new P(n,3)),o.setAttribute("kind",new P(s,1));const l=new S(o,L({uniforms:i,side:ie,vertexShader:`
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
          // by day warm-grey concrete in the low sun (it read as one more violet-grey slab beside the car)
          vec3 c = mix(vec3(.25, .21, .2), vec3(.16, .13, .18), uNight);
          c *= .7 + .3 * hash12(floor(vec2(vS * 2., vP.y * 6.)));
          // pink rim of the lamps along the top
          c += vec3(.5, .25, .4) * smoothstep(.6, .85, vP.y) * (1. - tun) * mix(.2, .5, uNight);
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
        }`}));return l.name="ride-barriers",t.add(l),t}function Wt(i){const t=[],a=(c,v,p,x,f,d,h)=>{const w=new te(c,v,p);w.translate(x,f,d),w.setAttribute("glow",new P(new Array(w.attributes.position.count).fill(h),1)),w.deleteAttribute("uv"),t.push(w)};a(.22,10,.22,0,5,0,0),a(2.8,.14,.14,-1.4,9.8,0,0),a(.9,.12,.38,-2.6,9.68,0,1);const o=qe(t),n=Math.ceil(760/Y),s=n*2,r=new me;r.index=o.index;for(const c of Object.keys(o.attributes))r.setAttribute(c,o.attributes[c]);const e=new Float32Array(s*3);for(let c=0;c<n;c++)e.set([he.parapet+.2,-c*Y+2,1],c*3),e.set([he.median,-c*Y-14,-1],(n+c)*3);r.setAttribute("slot",new ee(e,3)),r.instanceCount=s;const l=new S(r,L({uniforms:i,vertexShader:`
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
        }`}));return l.name="ride-lamps",l}function qe(i){const t=Object.keys(i[0].attributes),a=new de;let o=0;const n=[],s=Object.fromEntries(t.map(r=>[r,[]]));for(const r of i){const e=r;for(const l of t)s[l].push(...e.attributes[l].array);if(e.index)for(const l of e.index.array)n.push(l+o);else for(let l=0;l<e.attributes.position.count;l++)n.push(l+o);o+=e.attributes.position.count}for(const r of t)a.setAttribute(r,new P(s[r],i[0].attributes[r].itemSize));return a.setIndex(n),a}const je=720,lt=40,ne=-46;function _t(i){let t=i>>>0;return()=>{t=t+1831565813>>>0;let a=t;return a=Math.imul(a^a>>>15,a|1),a^=a+Math.imul(a^a>>>7,a|61),((a^a>>>14)>>>0)/4294967296}}const It=[{face:9.5,side:1,w:[9,18],d:[10,22],h:[22,95],row:0},{face:34,side:1,w:[12,26],d:[14,30],h:[45,170],row:1},{face:-25,side:-1,w:[9,18],d:[10,22],h:[22,95],row:0},{face:-52,side:-1,w:[12,26],d:[14,30],h:[45,170],row:1}];function $t(i){const t=_t(20261020),a=[],o=[];for(const n of It){if(n.row===1&&!i.midRow)continue;let s=0;for(;s<je-8;){const r=n.w[0]+t()*(n.w[1]-n.w[0]),e=n.d[0]+t()*(n.d[1]-n.d[0]),l=n.h[0]+Math.pow(t(),1.6)*(n.h[1]-n.h[0]),c=1.5+t()*(n.row?10:5),v=n.face+n.side*(e/2+t()*(n.row?14:3)),p=-(s+r/2);if(a.push({x:v,z:p,w:e,d:r,h:l,seed:t(),row:n.row,side:n.side}),n.row===0&&i.signs&&t()<.62){const x=1+(t()<.3?1:0);for(let f=0;f<x;f++){const d=v-n.side*(e/2)-n.side*1.1;o.push({x:d,z:p+r/2-.5-f*(r-1),y:7+t()*14,cell:Math.floor(t()*16),seed:t(),side:n.side})}}s+=r+c}}return{towers:a,signs:o}}function qt(i,t){const a=new te(1,1,1);a.translate(0,.5,0);const o=new me;o.index=a.index,o.setAttribute("position",a.attributes.position),o.setAttribute("normal",a.attributes.normal);const n=t.towers.length,s=new Float32Array(n*4),r=new Float32Array(n*4);t.towers.forEach((l,c)=>{s.set([l.x,l.z,l.w,l.d],c*4),r.set([l.h,l.seed,l.row,l.side],c*4)}),o.setAttribute("aBox",new ee(s,4)),o.setAttribute("aMeta",new ee(r,4)),o.instanceCount=n;const e=new S(o,L({uniforms:i,vertexShader:`
        attribute vec4 aBox;
        attribute vec4 aMeta;
        varying vec3 vP, vN, vL;
        varying float vSeed, vH, vS, vRow;
        void main() {
          float h = aMeta.x, seed = aMeta.y, row = aMeta.z, side = aMeta.w;
          float zc = wrapZ(aBox.y, ${je}., ${lt}.);
          float s = uD - zc;
          float bri = onBridge(s);
          float ms = mainStreet(s);
          // the main street crowds in (nearer, a little lower); on the bridge the towers fall away below the deck
          float x = aBox.x - side * ms * (row < .5 ? 2.5 : 10.);
          float hh = h * (1. - .25 * ms * (1. - row));
          vec3 p = position * vec3(aBox.z, hh - ${ne}., aBox.w);
          p.y += ${ne}.;
          p.y -= bri * (hh + 120.);
          p.x += x; p.z += zc;
          vL = position * vec3(aBox.z, hh - ${ne}., aBox.w) + vec3(0., ${ne}., 0.);
          // the seed on a 1/4096 grid (exact in float): the fragment shader snaps it back, see there
          vN = normal; vSeed = floor(seed * 4096.) / 4096.; vH = hh; vS = s; vRow = row; vP = p;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(bend(p), 1.);
        }`,fragmentShader:`
        varying vec3 vP, vN, vL;
        varying float vSeed, vH, vS, vRow;
        void main() {
          // a per-tower constant arrives through interpolation with a tiny error that the window hashes blow up: a
          // window whose hash sat near the lit threshold came out as per-pixel snow. Snapped back to its grid it is exact.
          float SEED = floor(vSeed * 4096. + .5) / 4096.;
          float side = abs(vN.x) > .5 ? 1. : 0.;
          float top = vN.y > .5 ? 1. : 0.;
          // facade coordinate across (u) and up (v)
          float u = side > .5 ? vL.z : vL.x;
          float v = vL.y;
          vec3 base = mix(vec3(.07, .06, .1), vec3(.13, .1, .15), hash11(SEED * 13.));
          // dusk: the walls take the low afterglow from ahead-left (the right-hand walls and the roofs catch it, the
          // faces toward the car stay in shade), so the near facades read darker than the haze behind them
          // (a low amber sun, strong enough that lit and shaded walls part: the shade keeps a cool sky fill, the lit
          // walls go warm-neutral instead of the shared violet-grey)
          vec3 sunL = normalize(vec3(-.55, .3, -.78));
          float sun = max(dot(vN, sunL), 0.) + .3 * top;
          vec3 dusk = base * .85 + vec3(.02, .025, .045) + vec3(.62, .42, .24) * sun;
          base = mix(dusk, base, uNight);
          // floors 3.4 m, bays 1.7 m between piers; windows lit by building, floor and bay, mostly warm
          vec2 g = vec2(u / 1.7, v / 3.4);
          vec2 cell = floor(g);
          vec2 f = fract(g);
          // past a few pixels a bay the panes and their on/off pattern blur to their average (no sparkle in the distance)
          vec2 fw = fwidth(g);
          float aa = clamp(1.6 - 2.4 * max(fw.x, fw.y), 0., 1.);
          float pane = mix(.33, step(.16, f.x) * step(f.x, .84) * step(.3, f.y) * step(f.y, .78), aa);
          float pier = mix(.16, 1. - step(.08, f.x) * step(f.x, .92), aa);
          float ratio = mix(.16, .30, uNight) * (.6 + .8 * hash11(SEED * 7.));
          // whole floors lit or dark together, as offices are
          float floorOn = step(.45, hash12(vec2(cell.y, SEED * 31.)));
          float lit = mix(ratio * .8, step(1. - ratio * (.55 + .7 * floorOn), hash12(cell + SEED * 91.)), aa);
          float tint = hash12(cell * 1.7 + SEED);
          vec3 wc = tint < .70 ? vec3(1., .86, .68) : tint < .95 ? vec3(.72, .82, .94) : tint < .98 ? vec3(1., .66, .82) : vec3(.66, .94, .82);
          wc *= mix(.45, .2 + 1.05 * pow(hash12(cell + 3.), 3.), aa);
          vec3 c = base * (1. + pier * .35);
          // by day a lit office is a faint warm square against glass that holds the sky
          c += wc * pane * lit * step(0., v) * 1.15 * mix(1., .48, vRow) * mix(.45, 1., uNight);
          c += mix(vec3(.17, .17, .22) * (.55 + .6 * sun), vec3(.025, .03, .045), uNight) * pane * (1. - lit);
          // neon strip lights along the floor slabs of some towers, every few floors
          float every = 3.4 * (3. + floor(hash11(SEED * 41.) * 4.));
          float dv = abs(fract(v / every + .5) - .5) * every; // metres from the nearest strip
          float strip = step(.6, hash11(SEED * 37.)) * smoothstep(.3, .08, dv) * step(4., v);
          c += mix(vec3(1., .25, .7), vec3(.25, .85, 1.), step(.5, hash11(SEED * 43.))) * strip * mix(.5, 1.2, uNight);
          // a big screen on some road-facing walls: slow colour bands and a scan line
          float screen = step(.82, hash11(SEED * 29.)) * side * step(14., v) * step(v, 14. + min(26., vH * .4)) * step(abs(u), 0.42 * 18.);
          vec3 sc = .5 + .5 * cos(6.28 * (vec3(0., .33, .67) + (v * .03 + uT * .05 + SEED)));
          // the scan line fades out once it is finer than a couple of pixels (it turned into snow on far screens)
          float scan = step(.5, fract(v * .8 - uT * 2.));
          scan = mix(.5, scan, clamp(1.5 - 2. * fwidth(v * .8), 0., 1.));
          sc = mix(sc, vec3(1., .3, .7), .35) * (.85 + .15 * scan);
          c = mix(c, sc * 1.6, screen * mix(.4, 1., uNight));
          // roof edge lamps on some towers
          float edge = smoothstep(vH - .6, vH - .1, v) * (1. - top) * step(.5, hash11(SEED * 3.));
          c += mix(vec3(1., .3, .6), vec3(.3, .8, 1.), step(.5, hash11(SEED * 17.))) * edge * 1.4;
          c = mix(c, base * .8, top);
          // city glow from below, haze with distance
          c += vec3(.22, .08, .18) * exp(-(v - ${ne}.) / 40.) * .6;
          float d = length(vP);
          c = mix(c, uFogColor, fogOf(d) * (1. - lit * pane * .15));
          // dusk air: the farther towers (and the back row) fade into a lighter haze, the near ones keep their shade
          float air = (1. - uNight) * (1. - exp(-d / 240.)) * (.5 + .25 * vRow);
          c = mix(c, vec3(.66, .53, .5), air);
          gl_FragColor = lin(c);
        }`}));return e.name="ride-towers",e}const jt=["HOTEL","BAR","24H","KTV","NOODLE","CAFE","ARCADE","NOCTILUCA","JINRUI","GOODBOAT","TAXI","BOOKS","DINER","CLUB","LIVE","OPEN"],tt=["#ff4fb8","#4fe3ff","#ffb547","#9a7bff","#ff5a5a","#5affc8","#ff7ae0","#7ad0ff"];function Xt(){const i=document.createElement("canvas");i.width=1024,i.height=1024;const t=i.getContext("2d");t.fillStyle="#000",t.fillRect(0,0,1024,1024),jt.forEach((o,n)=>{const s=n%8*128,r=Math.floor(n/8)*512,e=tt[n%tt.length];t.fillStyle="#0b0710",t.fillRect(s+10,r+10,108,492),t.strokeStyle=e,t.lineWidth=5,t.shadowColor=e,t.shadowBlur=16,t.strokeRect(s+16,r+16,96,480),t.fillStyle="#fff",t.font='bold 64px "Oxanium", "Big Shoulders Display", sans-serif',t.textAlign="center",t.textBaseline="middle";const l=o.split(""),c=Math.min(78,440/l.length);l.forEach((v,p)=>{t.shadowBlur=22,t.fillStyle=e,t.fillText(v,s+64,r+256-(l.length-1)*c/2+p*c),t.shadowBlur=0,t.fillStyle="#fff8",t.fillText(v,s+64,r+256-(l.length-1)*c/2+p*c)})});const a=new ze(i);return a.colorSpace=Pe,a.anisotropy=4,a}function Gt(i,t){if(!t.signs.length)return null;const a=new oe(1.7,6.8),o=new me;o.index=a.index,o.setAttribute("position",a.attributes.position),o.setAttribute("uv",a.attributes.uv);const n=t.signs.length,s=new Float32Array(n*4),r=new Float32Array(n*2);t.signs.forEach((c,v)=>{s.set([c.x,c.z,c.y,c.cell],v*4),r.set([c.seed,c.side],v*2)}),o.setAttribute("aSign",new ee(s,4)),o.setAttribute("aMeta",new ee(r,2)),o.instanceCount=n;const e=Xt(),l=new S(o,L({uniforms:{...i,uAtlas:{value:e}},transparent:!0,depthWrite:!1,blending:$e,side:ie,vertexShader:`
        attribute vec4 aSign;
        attribute vec2 aMeta;
        varying vec2 vUv;
        varying float vFl, vFog;
        void main() {
          float zc = wrapZ(aSign.y, ${je}., ${lt}.);
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
        }`}));return l.name="ride-signs",l.userData.textures=[e],l}function Vt(i){const t=new oe(2400,2400,1,1);t.rotateX(-Math.PI/2),t.translate(0,ne+6,-800);const a=new S(t,L({uniforms:i,vertexShader:`
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
          float d = length(vP - vec3(0., ${ne}., 0.));
          c = mix(c, uFogColor, fogOf(d) * .85);
          gl_FragColor = lin(c);
        }`}));return a.name="ride-ground",a.renderOrder=-8,a}const I=(i,t,a,o,n)=>i.push(...t,...a,...o,...t,...o,...n);function ct(i,t,a={}){return L({uniforms:i,side:ie,...a,vertexShader:`
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
      }`,fragmentShader:t})}function Yt(i,[t,a]){const o=[],n=[],s=[];for(let x=t;x<a;x+=4){const f=o.length;I(o,[-10.3,0,0],[-10.3,6.4,0],[-10.3,6.4,-4],[-10.3,0,-4]),I(o,[5,0,0],[5,0,-4],[5,6.4,-4],[5,6.4,0]),I(o,[-10.3,6.4,0],[5,6.4,0],[5,6.4,-4],[-10.3,6.4,-4]);for(let d=f;d<o.length;d+=3)n.push(x),s.push(d-f<36?0:1);if(Math.round((x-t)/4)%2===0)for(const d of[-10.3+.25,5-.25]){const h=o.length,w=6.4-.9,m=6.4-.6,u=-.4,y=-2.8,k=d+(d<0?.18:-.18);I(o,[k,w,u],[k,m,u],[k,m,y],[k,w,y]),I(o,[d,w,u],[k,w,u],[k,w,y],[d,w,y]);for(let z=h;z<o.length;z+=3)n.push(x),s.push(2)}}for(const[x,f]of[[t,1],[a,-1]]){const d=o.length,h=0;I(o,[-34,0,h],[-10.3,0,h],[-10.3,24,h],[-34,24,h]),I(o,[5,0,h],[30,0,h],[30,24,h],[5,24,h]),I(o,[-10.3,6.4,h],[5,6.4,h],[5,24,h],[-10.3,24,h]);for(let w=d;w<o.length;w+=3)n.push(x),s.push(3);if(f>0){const w=o.length;I(o,[-6,6.4+1.2,h+.05],[2.5,6.4+1.2,h+.05],[2.5,6.4+3,h+.05],[-6,6.4+3,h+.05]);for(let m=w;m<o.length;m+=3)n.push(x),s.push(4)}}const v=new de;v.setAttribute("position",new P(o,3)),v.setAttribute("aS",new P(n,1)),v.setAttribute("aK",new P(s,1));const p=new S(v,ct(i,`
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
    }`));return p.name="ride-tunnel",p}function Qt(i,[t,a]){const o=[],n=(f,d,h,w,m,u,y,k)=>{const z=new te(d-f,w-h,u-m);z.translate((f+d)/2,(h+w)/2,(m+u)/2),z.deleteAttribute("uv"),z.deleteAttribute("normal");const T=z.attributes.position.count;z.setAttribute("aS",new P(new Array(T).fill(y),1)),z.setAttribute("aK",new P(new Array(T).fill(k),1)),o.push(z)},s=t+170,r=a-190,e=92,l=-24,c=8;for(const f of[s,r]){for(const d of[l,c])n(d-1.6,d+1.6,-46,e,-1.6,1.6,f,0),n(d-.5,d+.5,e,e+1.2,-.5,.5,f,2);for(const d of[28,82])n(l,c,d,d+3,-1.2,1.2,f,0)}const v=f=>{if(f<s)return 2+(e-2)*Math.max(0,(f-t)/(s-t))**1.6;if(f>r)return 2+(e-2)*Math.max(0,(a-f)/(a-r))**1.6;const d=(f-s)/(r-s);return e-(e-10)*4*d*(1-d)};for(const f of[l+1,c-1]){for(let d=t;d<a;d+=6){const h=v(d),w=v(d+6),m=new de,u=[];I(u,[f-.35,h,0],[f+.35,h,0],[f+.35,w,-6],[f-.35,w,-6]),I(u,[f,h-.35,0],[f,h+.35,0],[f,w+.35,-6],[f,w-.35,-6]),m.setAttribute("position",new P(u,3)),m.setAttribute("aS",new P(new Array(12).fill(d),1)),m.setAttribute("aK",new P(new Array(12).fill(1),1)),o.push(m)}for(let d=t+12;d<a-6;d+=12){const h=v(d);h<3||n(f-.06,f+.06,1,h,-.06,.06,d,1)}}const p=qe(o.map(f=>f.index?f.toNonIndexed():f)),x=new S(p,ct(i,`
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
    }`));return x.name="ride-bridge",x}const Zt=[["BAY TUNNEL","HARBOUR  2 km"],["BAY BRIDGE","NOCTILUCA BAY"],["EXIT 1 km","MAIN STREET"],["MAIN STREET","KERB  ·  CENTRE"]];function Jt(){const i=document.createElement("canvas");i.width=1024,i.height=512;const t=i.getContext("2d");Zt.forEach(([o,n],s)=>{const r=s%2*512,e=Math.floor(s/2)*256;t.fillStyle="#0c5a4a",t.fillRect(r,e,512,256),t.strokeStyle="#e8f4ee",t.lineWidth=8,t.strokeRect(r+14,e+14,484,228),t.fillStyle="#f4faf6",t.textAlign="center",t.font='bold 64px "Oxanium", "Big Shoulders Display", sans-serif',t.fillText(o,r+256,e+110),t.font='44px "Oxanium", "Big Shoulders Display", sans-serif',t.fillText(n,r+256,e+186)});const a=new ze(i);return a.colorSpace=Pe,a.anisotropy=4,a}function eo(i,t){const a=[],o=[],n=[],s=[],r=(p,x,f,d)=>{a.push(...p),o.push(...x),n.push(f),s.push(d)},e=(p,x,f,d,h,w)=>{for(const m of[h,h-.5]){const u=[[p,f,m],[x,f,m],[x,d,m],[p,f,m],[x,d,m],[p,d,m]];for(const y of u)r(y,[0,0],w,0)}};t.forEach(({s:p,board:x})=>{e(-10,-9.6,0,8.2,0,p),e(4.6,5,0,8.2,0,p),e(-10,5,7.6,8.2,0,p);const f=x%2/2,d=1-(Math.floor(x/2)+1)/2,h=-6.8,w=.6,m=4.8,u=7.5,y=.06,k=[[h,m,y,f,d],[w,m,y,f+.5,d],[w,u,y,f+.5,d+.5],[h,m,y,f,d],[w,u,y,f+.5,d+.5],[h,u,y,f,d+.5]];for(const[z,T,Q,C,G]of k)r([z,T,Q],[C,G],p,1)});const l=new de;l.setAttribute("position",new P(a,3)),l.setAttribute("uv",new P(o,2)),l.setAttribute("aS",new P(n,1)),l.setAttribute("aK",new P(s,1));const c=Jt(),v=new S(l,L({uniforms:{...i,uBoards:{value:c}},side:ie,vertexShader:`
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
      }`}));return v.name="ride-gantries",v.userData.textures=[c],v}const dt=520,to=40;function oo(){const i=[],t=(a,o,n,s,r,e)=>{const l=new te(a,o,n);l.translate(0,s,r),l.deleteAttribute("uv"),l.setAttribute("kind",new P(new Array(l.attributes.position.count).fill(e),1)),i.push(l)};return t(1.82,.62,4.4,.62,0,0),t(1.5,.5,2.2,1.18,.25,1),qe(i)}function ao(){const i=[],t=[],a=[],o=(r,e,l,c,v,p)=>{const x=[[r,l,v],[e,l,v],[e,c,v],[r,l,v],[e,c,v],[r,c,v]],f=[[0,0],[1,0],[1,1],[0,0],[1,1],[0,1]];x.forEach((d,h)=>{i.push(...d),t.push(...f[h]),a.push(p)})},n=(r,e,l,c,v,p)=>{const x=[[r,l,c],[e,l,c],[e,l,v],[r,l,c],[e,l,v],[r,l,v]],f=[[0,0],[1,0],[1,1],[0,0],[1,1],[0,1]];x.forEach((d,h)=>{i.push(...d),t.push(...f[h]),a.push(p)})};o(-1.1,1.1,.35,.95,-2.22,0),o(-1.1,1.1,.5,1,2.22,1);for(const r of[-.7,.7]){const e=[[r,.55,-2.2],[r,.85,-2.2],[r,.85,18],[r,.55,-2.2],[r,.85,18],[r,.55,18]],l=[[0,0],[0,1],[1,1],[0,0],[1,1],[1,0]];e.forEach((c,v)=>{i.push(...c),t.push(...l[v]),a.push(2)})}n(-1.6,1.6,.03,-2.2,-14,3);const s=new de;return s.setAttribute("position",new P(i,3)),s.setAttribute("uv",new P(t,2)),s.setAttribute("kind",new P(a,1)),s}function no(i){const t=[];let a=7;const o=()=>(a=a*16807%2147483647)/2147483647,n=Math.round(i*.4);for(let s=0;s<i;s++){const r=s>=n,e=r?he.oncoming:[he.mid,he.far];t.push({x:e[Math.floor(o()*e.length)]+(o()-.5)*.4,z0:-o()*dt,v:r?18+o()*12:16+o()*10,dir:r?1:-1,paint:o()})}return t}const ot=`
  attribute vec4 aCar; // x, z0, speed, dir (+1 oncoming, -1 our way)
  attribute float aPaint;
  attribute float kind;
  varying vec3 vP;
  varying float vKind, vPaint, vRel, vDir, vS;
  varying vec2 vUv;
  void main() {
    float dir = aCar.w;
    // oncoming cars come toward us at their own speed plus ours; our way, they move ahead of us at theirs
    float z = wrapZ(aCar.y + dir * aCar.z * uT, ${dt}., ${to}.);
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
  }`;function at(i,t){const a=new me;i.index&&(a.index=i.index);for(const s of Object.keys(i.attributes))a.setAttribute(s,i.attributes[s]);const o=new Float32Array(t.length*4),n=new Float32Array(t.length);return t.forEach((s,r)=>{o.set([s.x,s.z0,s.v,s.dir],r*4),n[r]=s.paint}),a.setAttribute("aCar",new ee(o,4)),a.setAttribute("aPaint",new ee(n,1)),a.instanceCount=t.length,a}function io(i,t,{trails:a=!0}={}){const o=new K;o.name="ride-traffic";const n=new S(at(oo(),t),L({uniforms:i,vertexShader:ot,fragmentShader:`
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
        }`}));n.name="ride-traffic-bodies",o.add(n);const s=ao(),r=new S(at(s,t),L({uniforms:{...i,uTrails:{value:a?1:0}},transparent:!0,depthWrite:!1,blending:$e,side:ie,defines:{USE_UV:""},vertexShader:ot,fragmentShader:`
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
        }`}));return r.name="ride-traffic-lamps",r.renderOrder=3,o.add(r),o}const ve=new B(0,1.067,-1.142),_e=new B(0,1.325,-.565),so=1.78,ft=[-.78,.02],ut=1.75;function ro(i,t){if(!t)return null;const a=new oe(.018,.9),o=new me;o.index=a.index,o.setAttribute("position",a.attributes.position),o.setAttribute("uv",a.attributes.uv);const n=new Float32Array(t*4);let s=3;const r=()=>(s=s*16807%2147483647)/2147483647;for(let l=0;l<t;l++)n.set([(r()-.5)*30,r()*18,-r()*70,r()],l*4);o.setAttribute("aDrop",new ee(n,4)),o.instanceCount=t;const e=new S(o,L({uniforms:i,transparent:!0,depthWrite:!1,blending:$e,vertexShader:`
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
        }`}));return e.name="ride-rain",e}function lo(i,t){const a=ve.distanceTo(_e),o=new oe(so,a,1,1),n=new S(o,L({uniforms:{...i,uWipe:t.uniforms.uWipe,uWipeRate:t.uniforms.uWipeRate,uPivots:{value:new We(...ft)}},transparent:!0,depthWrite:!1,vertexShader:`
        varying vec2 vQ;
        void main() {
          vQ = vec2(position.x, position.y + ${(a/2).toFixed(4)});
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
          float a = clamp(ang / ${ut.toFixed(3)}, 0., 1.);
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
        }`})),s=_e.clone().sub(ve).normalize(),r=new B(1,0,0).cross(s).normalize();return n.position.copy(ve).addScaledVector(s,a/2).addScaledVector(r,-.008),n.quaternion.setFromRotationMatrix(new it().makeBasis(new B(1,0,0),s,r)),n.name="ride-windscreen-drops",n.renderOrder=5,n}function co(){const i=_e.clone().sub(ve).normalize(),t=new B(1,0,0).cross(i).normalize(),a=new it().makeBasis(new B(1,0,0),i,t),o=new K;o.name="ride-wipers";const n=[],s=new Me({color:723984,roughness:.6,metalness:.3});for(const r of ft){const e=new K;e.position.copy(ve).addScaledVector(t,-.025).add(new B(r,0,0)).addScaledVector(i,.02),e.quaternion.setFromRotationMatrix(a);const l=new K;e.add(l);const c=new S(new te(.72,.016,.014),s);c.position.set(.36,0,0);const v=new S(new te(.66,.026,.02),s);v.position.set(.4,0,-.008),l.add(c,v),o.add(e),n.push(l)}return{group:o,blades:n,material:s}}const A={x:.36,z0:-.36,z1:.38,y:1.33};function fo(i){i.updateMatrixWorld(!0);const t=new st,a=new B,o=new B;i.traverse(n=>{if(!n.isMesh||n.name)return;const s=n.material;s!=null&&s.isMeshBasicMaterial&&s.transparent&&(t.setFromObject(n).getSize(a),t.getCenter(o),a.x<.05&&a.z>2.4&&a.z<2.8&&Math.abs(o.x)>.7&&o.y>1&&o.y<1.4&&(n.name=o.x>0?"glass-right":"glass-left"))})}function uo(i,t,a){let o=null,n=1/0;const s=new B;return i.traverse(r=>{if(r.name!==t)return;r.getWorldPosition(s);const e=a?s.distanceTo(a):0;e<n&&(n=e,o=r)}),o}function ho(i,t){const a=[],o=new Map,n=new st;return i.updateMatrixWorld(!0),i.traverse(s=>{if(!s.isMesh||s.userData.ride||s.userData.noCollide||[].concat(s.material).some(e=>e==null?void 0:e.isShaderMaterial)||(n.setFromObject(s),n.max.y<A.y||n.min.x>A.x||n.max.x<-.36||n.min.z>A.z1||n.max.z<A.z0))return;const r=e=>{if(!e||e.transparent)return e;if(!o.has(e)){const l=e.clone();l.onBeforeCompile=c=>{c.uniforms.uRoofOpen=t,c.vertexShader=c.vertexShader.replace("void main() {",`varying vec3 vRoofW;
void main() {`).replace("#include <project_vertex>",`#include <project_vertex>
  vRoofW = (modelMatrix * vec4(transformed, 1.0)).xyz;`),c.fragmentShader=c.fragmentShader.replace("void main() {",`uniform float uRoofOpen;
varying vec3 vRoofW;
void main() {
  if (uRoofOpen > 0.001 && vRoofW.y > ${A.y.toFixed(3)} && abs(vRoofW.x) < ${A.x.toFixed(3)} && vRoofW.z > ${A.z0.toFixed(3)} && vRoofW.z < ${A.z0.toFixed(3)} + ${(A.z1-A.z0).toFixed(3)} * uRoofOpen) discard;`)},l.customProgramCacheKey=()=>"ride-roof",o.set(e,l),a.push(l)}return o.get(e)};s.material=Array.isArray(s.material)?s.material.map(r):r(s.material)}),a}function vo(){const i=document.createElement("canvas");i.width=512,i.height=224;const t=new ze(i);t.colorSpace=Pe;const a=new S(new oe(.31,.13),new pe({map:t,toneMapped:!1}));a.position.set(-.012,.983,-.9115),a.rotation.x=-.08,a.name="ride-screen";let o="";return{mesh:a,draw:s=>{const r=JSON.stringify(s);if(r===o)return;o=r;const e=i.getContext("2d");e.fillStyle="#050b14",e.fillRect(0,0,512,224),e.fillStyle="rgba(120,170,230,0.5)",e.fillRect(12,11,488,1),e.font="11px sans-serif",e.fillStyle="#7fa8d8",e.fillText(s.clock,15,25),e.fillText("MEDIA   ROUTE   CAR",190,25);const l=s.station>0;if(e.fillStyle=l?"#13243a":"#0b1420",e.fillRect(14,38,200,172),e.fillStyle=l?"#ff9ab8":"#4a6a8a",e.font="bold 15px sans-serif",e.fillText(ke[s.station].name,24,62),e.font="12px sans-serif",e.fillStyle="#9cc4ee",l){e.fillText(ke[s.station].track,24,84),e.fillText(ke[s.station].artist,24,102);for(let c=0;c<14;c++){const v=8+(c*37+s.beat*13)%40;e.fillStyle="#4aa8ff",e.fillRect(24+c*13,190-v,8,v)}}else e.fillText("F at the screen: tune in",24,84);e.strokeStyle="#2a4a6a",e.lineWidth=4,e.beginPath(),e.moveTo(250,120),e.lineTo(490,120),e.stroke(),e.strokeStyle="#4aa8ff",e.beginPath(),e.moveTo(250,120),e.lineTo(250+240*s.progress,120),e.stroke(),rt.forEach((c,v)=>{const p=250+240*c.t/$;e.fillStyle=s.chapter>=c.n?"#ff9ab8":"#2a4a6a",e.beginPath(),e.arc(p,120,6,0,Math.PI*2),e.fill(),e.fillStyle="#7fa8d8",e.font="10px sans-serif",e.fillText(String(v+1),p-3,142)}),e.fillStyle="#ffffff",e.beginPath(),e.arc(250+240*s.progress,120,8,0,Math.PI*2),e.fill(),s.done?(e.strokeStyle="#ff9ab8",e.lineWidth=3,e.beginPath(),e.moveTo(252,72),e.lineTo(259,80),e.lineTo(273,62),e.stroke(),e.fillStyle="#ff9ab8",e.font="bold 22px sans-serif",e.fillText(s.title,282,80,216)):(e.fillStyle="#9cc4ee",e.font="bold 16px sans-serif",e.fillText(s.title,250,80,248)),e.fillStyle="#9cc4ee",e.font="14px sans-serif",e.fillText(s.eta,250,178,248),t.needsUpdate=!0},tex:t}}function po(){const i=document.createElement("canvas");i.width=512,i.height=224;const t=new ze(i);t.colorSpace=Pe;const a=new S(new oe(.33,.13),new pe({map:t,toneMapped:!1}));a.position.set(-.39,.985,-.9115),a.rotation.x=-.08,a.name="ride-cluster";let o="";return{mesh:a,draw:(s,r)=>{const e=`${Math.round(s)}${r}`;if(e===o)return;o=e;const l=i.getContext("2d");l.setTransform(.5,0,0,.5,0,0),Ft(l,1024,448,Math.round(s),r),t.needsUpdate=!0},tex:t}}function mo(){const i=new K;i.name="ride-charm";const t=new S(new Ie(.0012,.0012,.11,4),new pe({color:10127984}));t.position.y=-.055;const a=new zt;a.absarc(0,0,.032,Math.PI*.15,Math.PI*1.85,!1),a.absarc(.014,0,.026,Math.PI*1.75,Math.PI*.25,!0);const o=new S(new Pt(a,{depth:.003,bevelEnabled:!1,curveSegments:18}),new Me({color:15918792,emissive:9069624,emissiveIntensity:.5,roughness:.9,side:ie}));return o.position.set(0,-.14,-.0015),o.rotation.z=.25,i.add(t,o),i.position.set(0,1.262,-.63),i}const go=[{rough:.38,set:{roughness:.74,color:4605003}},{rough:.48,set:{roughness:.64,color:3420986}},{rough:.5,set:{roughness:.7,color:2763055}}];function wo(i){const t=new Map;return i.traverse(a=>{const o=a.material;if(!a.isMesh||!(o!=null&&o.isMeshStandardMaterial)||o.metalness>.05)return;const n=go.find(s=>Math.abs(o.roughness-s.rough)<.015);if(n){if(!t.has(o)){const s=o.clone();s.roughness=n.set.roughness,s.color.setHex(n.set.color),t.set(o,s)}a.material=t.get(o)}}),[...t.values()]}function xo(){const i=new K;i.name="ride-lights";const t=new Mt(4868704,2102296,.34),a=new Be(16758984,1.3);a.position.set(20,30,-40),a.target.position.set(0,0,-10),a.castShadow=!0,a.shadow.mapSize.set(2048,2048),Object.assign(a.shadow.camera,{left:-30,right:30,top:30,bottom:-30,near:1,far:120}),a.shadow.bias=-4e-4,a.shadow.normalBias=.03;const o=new Fe(16761032,60,16,1.4);o.position.set(1.5,7,-20);const n=new Fe(16730832,30,9,1.6);n.position.set(-7,2,-12);const s=new Be(10131104,.52);s.position.set(-.2,1.7,1),s.target.position.set(.1,.85,-1);const r=new Be(16042144,0);r.position.set(-1.6,1.55,-.9),r.target.position.set(.5,.95,.2);const e=new Fe(8688832,.1,.8,2);return e.position.set(-.2,.95,-.6),i.add(t,a,a.target,o,n,s,s.target,r,r.target,e),{group:i,hemi:t,key:a,pool:o,mag:n,fill:s,win:r,disp:e}}function bo(i,t){const a=new K;a.name="ride-cabin";const o={value:0},n=ho(i,o),s=i.getObjectByName("sunroof");s&&(s.visible=!1);const r=new Me({color:11845078,roughness:.32,metalness:.35,emissive:1316900}),e=new K;e.name="ride-sunroof";const l=A.x*2,c=A.z1-A.z0,v=(A.z0+A.z1)/2;for(const[b,F,V,se]of[[l+.06,.03,0,A.z0-.015],[l+.06,.03,0,A.z1+.015],[.03,c,-.36-.015,v],[.03,c,A.x+.015,v]]){const xe=new S(new te(b,.012,F),r);xe.position.set(V,1.392,se),e.add(xe)}const p=new pe({color:3816026,transparent:!0,opacity:.35,depthWrite:!1,side:ie}),x=new S(new oe(l,c),p);x.rotation.x=Math.PI/2,x.position.set(0,1.455,v),a.add(e,x);const f=mo();a.add(f);const d=vo();a.add(d.mesh);const h=po();a.add(h.mesh);const w=new pe({color:3866522,toneMapped:!1}),m=new Me({color:2764344,roughness:.45,metalness:.5}),u=new K;u.name="ride-lightswitch",u.position.set(-.72,.79,-.858);const y=new S(new Ie(.024,.026,.022,20).rotateX(Math.PI/2),m);y.position.z=.011;const k=new S(new te(.004,.018,.004),w);k.position.set(0,.01,.023),y.add(k);const z=new S(new St(.005,12),w);z.position.set(.04,.012,.002),u.add(y,z),a.add(u),n.push(r,p,w,m,d.mesh.material,d.tex,h.mesh.material,h.tex);for(const b of[f,e,x,d.mesh,h.mesh,u])b.traverse(F=>{F.userData.ride=!0,F.raycast=()=>{}});a.traverse(b=>{b.userData.noCollide=!0});const T=i.getObjectByName("wheel"),Q=T==null?void 0:T.quaternion.clone();T&&n.push(...wo(T));const C=uo(i,"glass-right",new B(1,1.2,0)),G=(C==null?void 0:C.position.y)??0,W=i.getObjectByName("leadcar"),ge=W==null?void 0:W.position.clone(),we=new B(0,0,1),Te=new kt,Z={t0:-99,amp:0};return{group:a,owned:n,uniforms:{uRoofOpen:o},flick(b){Z.t0=b,Z.amp=.75},screen:d,update(b,F){o.value=b.roof,h.draw(b.speed*3.6,b.t>$-.5?"P":"D"),x.position.z=v+c*b.roof*.98,x.visible=b.roof>.01,C&&(C.position.y=G-.44*b.window),T&&T.quaternion.copy(Q).multiply(Te.setFromAxisAngle(we,D.clamp(-b.curve*900,-.5,.5)));const V=F-Z.t0,se=Z.amp*Math.exp(-V/2.2)*Math.sin(V*5.2);f.rotation.x=se+b.accel*.03+Math.sin(F*1.7)*.015*Math.min(1,b.speed/10),f.rotation.z=D.clamp(b.curve*b.speed*b.speed*.5,-.4,.4)+se*.2,w.color.setHex(b.head?3866522:1058330),y.rotation.z=b.head?-.9:0,W&&(W.position.x=ge.x-1+Math.sin(F*.31)*.12,W.position.z=ge.z-Math.max(0,b.t-($-14))**2*.9,W.visible=W.position.z>-400)},dispose(){var b;for(const F of n)(b=F.dispose)==null||b.call(F)}}}const nt={high:{traffic:46,rain:1400,signs:!0,midRow:!0,trails:!0},medium:{traffic:34,rain:800,signs:!0,midRow:!0,trails:!0},low:{traffic:24,rain:360,signs:!0,midRow:!1,trails:!1},software:{traffic:14,rain:0,signs:!1,midRow:!1,trails:!1}},Se={from:"kouka",title:"Kouka",text:"Saw your car come out of the bay tunnel on the traffic feed. The water is glowing again tonight, the plankton bloom. Look left before the bridge ends."},Ke=i=>{try{return new URLSearchParams(location.search).get(i)}catch{return null}};function yo(i){var a,o;const t=i.getObjectByName("leadcar");t&&t.parent!==i&&i.attach(t);for(const n of["road","wet","streaks","lights","skyDome"]){const s=i.getObjectByName(n);s&&(s.visible=!1)}i.traverse(n=>{var s;(s=n.userData)!=null&&s.update&&/mist/i.test(n.name||"")&&n.parent===i&&(n.visible=!1)});for(const n of i.children)n.isMesh&&((o=(a=n.material)==null?void 0:a.uniforms)!=null&&o.uSpeed)&&n!==t&&(n.visible=!1)}function Po(i,{quality:t}={}){const a=le.tier||(t==="low"?"low":"high"),o=nt[a]||nt.high,n=Ke("tod")?Ke("tod")!=="day":Tt(),s=Ke("rain")!=="0",r=i.root;fo(r),yo(r),i.walk={seated:!0,bounds:[-.3,-.3,.3,.3],eye:1.2,floorY:()=>0,colliders:[],start:{position:[0,1.2,.02],target:[0,1.06,-1]},fov:64},i.env=i.env||{},i.env.fog=n?{color:1709104,density:.0026}:{color:6178386,density:.0024},i.env.background=n?395278:3811908;const e=Nt();e.uTunnel.value.set(...O.tunnel),e.uBridge.value.set(...O.bridge),e.uRamp.value=O.ramp,e.uEnd.value=O.end,e.uNight.value=n?1:0,e.uFogColor.value.setHex(i.env.fog.color,At),e.uFogDensity.value=i.env.fog.density;const l=new K;l.name="ride";const c=$t({midRow:o.midRow,signs:o.signs}),v=[],p=g=>{g&&(l.add(g),v.push(...g.userData.textures||[]))};p(Lt(e)),p(Vt(e)),p(Kt(e)),p(Wt(e)),p(qt(e,c)),p(Gt(e,c)),p(Yt(e,O.tunnel)),p(Qt(e,O.bridge)),p(eo(e,[{s:420,board:0},{s:O.tunnel[0]-170,board:0},{s:O.bridge[0]+90,board:1},{s:O.bridge[1]-260,board:2},{s:O.ramp+60,board:3}])),p(io(e,no(o.traffic),{trails:o.trails})),p(ro(e,s?o.rain:0)),et(l),r.add(l);const x={uniforms:{uWipe:{value:0},uWipeRate:{value:0}}},f=bo(r),d=et(lo(e,x)),h=co();h.group.traverse(g=>{g.raycast=()=>{},g.userData.noCollide=!0});const w=xo();r.add(f.group,d,h.group,w.group);const m=Et(),u={t:0,chapter:1,station:0,roof:0,window:0,wipers:0,head:!0,arrived:!1,rain:0},y={roof:0,window:0},k=new Set,z=new Set,T=g=>{for(const R of z)try{R(g)}catch(N){console.error(N)}};let Q=performance.now(),C=0,G=!1;const W=()=>{Q=performance.now()};document.addEventListener("visibilitychange",W);const ge=(g,R={})=>T({kind:"toast",text:g,...R}),we={"radio:next":()=>{m.arm();const g=m.setStation(m.stationIndex+1===ke.length?0:m.stationIndex+1);return u.station=m.stationIndex,T({kind:"radio",station:u.station}),g.url?`${g.name} · "${g.track}", ${g.artist}`:"The radio clicks off. Just the road now."},"radio:off":()=>(m.arm(),m.setStation(0),u.station=0,T({kind:"radio",station:0}),"The radio clicks off. Just the road now."),"sunroof:toggle":()=>(m.arm(),m.click(),y.roof=y.roof>.5?0:1,y.roof?"The sunroof slides back. Cold air, and the towers leaning in overhead.":"The sunroof slides shut. The city sound drops away."),"window:toggle":()=>(m.arm(),m.click(),y.window=y.window>.5?0:1,y.window?"The window winds down. The wind comes in loud, smelling of rain and the sea.":"The window winds up. Quiet again."),"wipers:cycle":()=>(m.arm(),m.click("wiper"),u.wipers=(u.wipers+1)%3,u.wipers===0&&(G=!0),["Wipers off.","Wipers on.","Wipers fast."][u.wipers]+(u.rain<.05&&u.wipers?" The glass is dry; they squeak once and keep time.":"")),"lights:toggle":()=>(m.arm(),m.click(u.head?"off":"switch"),u.head=!u.head,u.head?"Headlamps on. The lane comes back out of the dark.":"Headlamps off. Between the street lamps the road goes dark; the car does not mind."),"charm:flick":()=>(f.flick(Te()),"You flick the paper moon. It spins on its thread, and the dashboard light runs around its edge."),"route:show":()=>{const g=Ze(u.t),R=Math.max(0,Math.round($-u.t));return u.arrived?"Main street, at the kerb. The route screen says: you have arrived.":`Chapter ${g.n} of 3, ${g.title}. Main street in ${Math.floor(R/60)}:${String(R%60).padStart(2,"0")}.`},"ride:restart":()=>(Z.seek(0),k.clear(),"The car pulls away from the kerb again."),horn:()=>(m.arm(),m.click(),null)},Te=()=>performance.now()/1e3,Z={get state(){return{...u,length:He,end:$}},message:Se,subscribe(g){return z.add(g),()=>z.delete(g)},use(g){var H,E,j,M;const R=(j=(E=(H=g==null?void 0:g.interactable)==null?void 0:H.primary)==null?void 0:E.effect)==null?void 0:j.device;if(!R||!we[R])return!1;const q=we[R]()||((M=g.lines)==null?void 0:M[0])||null;return q&&ge(q,{device:R}),!0},seek(g){u.t=Math.max(0,Math.min(He,g)),Q=performance.now(),g<Ne.mid&&k.delete("mid"),g<$&&(k.delete("arrive"),u.arrived=!1)},audio:m};i.ride=Z;const b={pos:new B,rx:0,ry:0,rz:0,on:!1,p0:new B,r0:new Rt},F=(g,R,N,q)=>{const H=performance.now(),E=Math.min(1,(H-Q)/1e3);Q=H;const j=document.hidden||!!le.hold||!!le.settling;j||(u.t=Math.min(He,u.t+E));const M=u.t,_=Dt(M),X=Le(M),ae=Ct(M),Re=(Le(M+.25)-Le(M-.25))/.5;e.uT.value=H/1e3%3600,e.uD.value=_,e.uSpeed.value=X,e.uCurve.value=ae,e.uHill.value=Ot(M),e.uHeading.value=Bt(M),u.rain=s?D.clamp((M-Ne.rain)/10,0,1)*(M>$+2?.8:1):0,e.uRain.value=u.rain,e.uHead.value=D.lerp(e.uHead.value,u.head?1:0,Math.min(1,E*6)),u.roof+=D.clamp(y.roof-u.roof,-E*.7,E*.7),u.window+=D.clamp(y.window-u.window,-E*.6,E*.6);const Ee=[0,.75,1.35][u.wipers];if(Ee)C+=E*Ee,G=!1;else if(G){const J=Math.ceil(C);C=Math.min(J,C+E*.75),C>=J&&(G=!1)}x.uniforms.uWipe.value=C,x.uniforms.uWipeRate.value=Ee||(G?.75:0);const De=C%1,ht=De<.5?De*2:2-De*2;for(const J of h.blades)J.rotation.z=ht*ut;f.update({t:M,roof:u.roof,window:u.window,curve:ae,speed:X,accel:Re,head:u.head},H/1e3);const re=Ze(M),Ce=Math.max(0,Math.round($-M)),be=u.arrived||Ce===0;f.screen.draw({station:u.station,beat:u.station?Math.floor(H/500):0,chapter:re.n,done:be,title:be?"Arrived":`${re.n} · ${re.title}`,progress:be?1:Math.min(1,M/$),eta:be?"Main street, at the kerb":`Main street ${Math.floor(Ce/60)}:${String(Ce%60).padStart(2,"0")}`,clock:new Date().toTimeString().slice(0,5)});const U=_>O.tunnel[0]&&_<O.tunnel[1]?1:0,fe=_>O.bridge[0]+40&&_<O.bridge[1]-40?1:0,vt=_%Y/Y,ye=Math.exp(-(((vt-.5)*Y/7)**2)),Xe=Math.exp(-(((_%8-4)/2.2)**2)),{key:Ge,hemi:Oe,pool:pt,mag:Ve,fill:Ye,win:mt}=w;Ge.intensity=U?.6+.9*Xe:(fe?.95:1.1)+.35*ye,Ge.color.setHex(U?16754768:fe?12110079:n?16758984:16765620),Oe.intensity=U?.2+.35*Xe:n?fe?.42:.3+.22*ye:.5+.1*ye,Oe.color.setHex(U?16752736:n?4999256:8288392),mt.intensity=n||U?0:.75,Oe.groundColor.setHex(fe?932410:U?3810320:2102296),pt.intensity=(U?25:60)*(u.head?1:.35);const gt=Math.floor(_/18)%3;Ve.color.setHex(U?16748608:fe?4247776:[16730832,4905215,16742970][gt]),Ve.intensity=U?10:30;const wt=(n?.52:.85)*(U?.85:le.still?1:.94+.16*ye);if(Ye.intensity=D.lerp(Ye.intensity,wt,Math.min(1,E*8)),m.update({speed:X,wind:Math.max(u.window,u.roof*.6),rain:u.rain,tunnel:U,paused:j}),q!=null&&q.warp&&se(q.warp,{v:X,accel:Re,tun:U,D:_,camera:N}),(re.n!==u.chapter||!k.has("chapter1"))&&(u.chapter=re.n,k.add("chapter1"),T({kind:"chapter",chapter:re})),!k.has("start")&&M>2.5&&(k.add("start"),Ue("ride:start",{place:"car-1",chapters:rt.length,length:$})),!k.has("mid")&&M>=Ne.mid&&(k.add("mid"),Ue("ride:mid",{place:"car-1",where:"bay-bridge",hie:Se.from,message:Se.text}),T({kind:"message",message:Se})),!k.has("arrive")&&M>=$&&(k.add("arrive"),u.arrived=!0,Ue("ride:arrive",{place:"car-1",to:"street-1"}),T({kind:"arrive"})),N&&!le.still){const J=Math.min(1,X/20);b.pos.set(Math.sin(H*.0021)*.002*J,(Math.sin(H*.0083)*.0022+Math.sin(H*.0191)*.0011)*J+(xe(H)-.5)*6e-4*J,0),b.rz=D.clamp(-ae*X*X*.035,-.035,.035),b.ry=D.clamp(ae*18,-.03,.03),b.rx=D.clamp(-Re*.004,-.02,.02),b.p0.copy(N.position),b.r0.copy(N.rotation),N.position.add(b.pos),N.rotation.x+=b.rx,N.rotation.y+=b.ry,N.rotation.z+=b.rz,b.on=!0}},V=new B,se=(g,{v:R,accel:N,tun:q,D:H,camera:E})=>{const j=le.still||a==="software"||!E,M=D.smoothstep(R,17,25),_=D.clamp(N/3,0,1),X=H-O.tunnel[1];if(g.amount=j?0:Math.min(.85,.5*M+.3*_+.25*q),g.lines=j?0:Math.min(.6,D.smoothstep(N,.3,1)*M+.35*q),g.flash=j||X<0?0:.8*Math.exp(-X/10),g.near=2.5,!j){const ae=D.smoothstep(-E.getWorldDirection(V).z,.35,.75);g.amount*=ae,g.lines*=ae,V.set(0,1.2,-100).project(E),g.center[0]=D.clamp(V.x*.5+.5,0,1),g.center[1]=D.clamp(V.y*.5+.5,0,1)}},xe=g=>{const R=Math.sin(g*12.9898)*43758.5453;return R-Math.floor(R)};i.update=F,i.afterRender=g=>{b.on&&(g.position.copy(b.p0),g.rotation.copy(b.r0),b.on=!1)};const Ae=i.dispose;return i.dispose=()=>{Ae==null||Ae(),document.removeEventListener("visibilitychange",W),m.dispose(),f.dispose(),z.clear(),l.traverse(g=>{g.isMesh&&(g.geometry.dispose(),g.material.dispose())});for(const g of v)g.dispose();d.geometry.dispose(),d.material.dispose(),h.material.dispose()},Z}export{Po as attachRide};
