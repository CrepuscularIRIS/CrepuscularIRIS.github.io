import{G as V,e as k,O as ht,U as Ue,a0 as qe,Z as ce,$ as z,a5 as Se,a6 as oe,P as fe,X as ae,L as le,C as Me,S as ke,y as et,d as tt,F as J,a4 as Ke,M as pe,ac as vt,bA as $,ao as pt,bB as C,R as ut,T as De,Q as Oe,a2 as mt,ak as gt,bC as ye,bD as ot,c as ie,bE as wt,bF as O,bG as xt,as as bt,bH as Be,bI as Fe,bJ as yt,bK as St,bL as Ne,bM as Mt,bN as kt,bO as Pt,bP as We,bQ as He}from"./index-C8VfKPqX.js";import{am as L,an as zt,ao as At,ap as Ve,aq as Tt,ar as Et,as as Rt}from"./texlocal-DrGh-ZXl.js";import{paintCluster as Ct}from"./ctex-DZjaDTL2.js";import"./react-three-fiber.esm-Y9WmF9rR.js";function Dt(i){const t=new V;t.name="ride-sky";const o=new k(new ht(900,32,16),L({uniforms:i,side:qe,depthWrite:!1,vertexShader:`
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
        }`}));o.renderOrder=-10,t.add(o);const a=new k(new Ue(650,650,260,96,1,!0),L({uniforms:i,side:qe,transparent:!0,depthWrite:!1,vertexShader:`
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
        }`}));return a.renderOrder=-9,t.add(a),t}const ve={mid:-3.6,far:-7.2,oncoming:[-12.2,-15.8,-19.2],parapet:4.6,median:-9.8},W=32,re=760,he=14;function Ot(i,t,o=190){const a=new fe(t-i,re,1,o);return a.rotateX(-Math.PI/2),a.translate((i+t)/2,0,he-re/2),a}function Bt(i){const t=new V;t.name="ride-road";const o=new k(Ot(-22,5.2),L({uniforms:i,vertexShader:`
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
          float lp = mod(s, ${W}.), lq = mod(s + 16., ${W}.);
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
        }`}));o.name="ride-road-deck",t.add(o);const a=new ce,n=[],s=[],r=(c,p,u,x)=>{for(let f=0;f<190;f++){const v=he-re*f/190,w=he-re*(f+1)/190;n.push(c,p,v,c,u,v,c,p,w,c,u,v,c,u,w,c,p,w);for(let m=0;m<6;m++)s.push(x)}},e=(c,p,u,x)=>{for(let f=0;f<190;f++){const v=he-re*f/190,w=he-re*(f+1)/190;n.push(c,u,v,p,u,v,c,u,w,p,u,v,p,u,w,c,u,w);for(let m=0;m<6;m++)s.push(x)}};r(-9.4,0,.85,0),r(-10.2,0,.85,0),e(-10.2,-9.4,.85,1),r(4.4,0,1.15,2),e(4.4,4.9,1.15,3),r(4.9,-1.2,1.15,2),r(-22,-1.2,.9,2),r(5.2,-1.4,0,4),a.setAttribute("position",new z(n,3)),a.setAttribute("kind",new z(s,1));const l=new k(a,L({uniforms:i,side:ae,vertexShader:`
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
        }`}));return l.name="ride-barriers",t.add(l),t}function Ft(i){const t=[],o=(c,p,u,x,d,f,v)=>{const w=new le(c,p,u);w.translate(x,d,f),w.setAttribute("glow",new z(new Array(w.attributes.position.count).fill(v),1)),w.deleteAttribute("uv"),t.push(w)};o(.22,10,.22,0,5,0,0),o(2.8,.14,.14,-1.4,9.8,0,0),o(.9,.12,.38,-2.6,9.68,0,1);const a=Ie(t),n=Math.ceil(760/W),s=n*2,r=new Se;r.index=a.index;for(const c of Object.keys(a.attributes))r.setAttribute(c,a.attributes[c]);const e=new Float32Array(s*3);for(let c=0;c<n;c++)e.set([ve.parapet+.2,-c*W+2,1],c*3),e.set([ve.median,-c*W-14,-1],(n+c)*3);r.setAttribute("slot",new oe(e,3)),r.instanceCount=s;const l=new k(r,L({uniforms:i,vertexShader:`
        attribute float glow;
        attribute vec3 slot;
        varying float vGlow, vS;
        varying vec3 vP;
        void main() {
          vec3 p = position;
          p.x *= slot.z;
          float z = wrapZ(slot.y, ${n*W}., 14.);
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
        }`}));return l.name="ride-lamps",l}function Ie(i){const t=Object.keys(i[0].attributes),o=new ce;let a=0;const n=[],s=Object.fromEntries(t.map(r=>[r,[]]));for(const r of i){const e=r;for(const l of t)s[l].push(...e.attributes[l].array);if(e.index)for(const l of e.index.array)n.push(l+a);else for(let l=0;l<e.attributes.position.count;l++)n.push(l+a);a+=e.attributes.position.count}for(const r of t)o.setAttribute(r,new z(s[r],i[0].attributes[r].itemSize));return o.setIndex(n),o}const $e=720,at=40,te=-46;function Nt(i){let t=i>>>0;return()=>{t=t+1831565813>>>0;let o=t;return o=Math.imul(o^o>>>15,o|1),o^=o+Math.imul(o^o>>>7,o|61),((o^o>>>14)>>>0)/4294967296}}const Ht=[{face:9.5,side:1,w:[9,18],d:[10,22],h:[22,95],row:0},{face:34,side:1,w:[12,26],d:[14,30],h:[45,170],row:1},{face:-25,side:-1,w:[9,18],d:[10,22],h:[22,95],row:0},{face:-52,side:-1,w:[12,26],d:[14,30],h:[45,170],row:1}];function Lt(i){const t=Nt(20261020),o=[],a=[];for(const n of Ht){if(n.row===1&&!i.midRow)continue;let s=0;for(;s<$e-8;){const r=n.w[0]+t()*(n.w[1]-n.w[0]),e=n.d[0]+t()*(n.d[1]-n.d[0]),l=n.h[0]+Math.pow(t(),1.6)*(n.h[1]-n.h[0]),c=1.5+t()*(n.row?10:5),p=n.face+n.side*(e/2+t()*(n.row?14:3)),u=-(s+r/2);if(o.push({x:p,z:u,w:e,d:r,h:l,seed:t(),row:n.row,side:n.side}),n.row===0&&i.signs&&t()<.62){const x=1+(t()<.3?1:0);for(let d=0;d<x;d++){const f=p-n.side*(e/2)-n.side*1.1;a.push({x:f,z:u+r/2-.5-d*(r-1),y:7+t()*14,cell:Math.floor(t()*16),seed:t(),side:n.side})}}s+=r+c}}return{towers:o,signs:a}}function Kt(i,t){const o=new le(1,1,1);o.translate(0,.5,0);const a=new Se;a.index=o.index,a.setAttribute("position",o.attributes.position),a.setAttribute("normal",o.attributes.normal);const n=t.towers.length,s=new Float32Array(n*4),r=new Float32Array(n*4);t.towers.forEach((l,c)=>{s.set([l.x,l.z,l.w,l.d],c*4),r.set([l.h,l.seed,l.row,l.side],c*4)}),a.setAttribute("aBox",new oe(s,4)),a.setAttribute("aMeta",new oe(r,4)),a.instanceCount=n;const e=new k(a,L({uniforms:i,vertexShader:`
        attribute vec4 aBox;
        attribute vec4 aMeta;
        varying vec3 vP, vN, vL;
        varying float vSeed, vH, vS, vRow;
        void main() {
          float h = aMeta.x, seed = aMeta.y, row = aMeta.z, side = aMeta.w;
          float zc = wrapZ(aBox.y, ${$e}., ${at}.);
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
          c += vec3(.22, .08, .18) * exp(-(v - ${te}.) / 40.) * .6;
          float d = length(vP);
          c = mix(c, uFogColor, fogOf(d) * (1. - lit * pane * .15));
          // dusk air: the farther towers (and the back row) fade into a lighter haze, the near ones keep their shade
          float air = (1. - uNight) * (1. - exp(-d / 240.)) * (.5 + .25 * vRow);
          c = mix(c, vec3(.66, .53, .5), air);
          gl_FragColor = lin(c);
        }`}));return e.name="ride-towers",e}const Ut=["HOTEL","BAR","24H","KTV","NOODLE","CAFE","ARCADE","NOCTILUCA","JINRUI","GOODBOAT","TAXI","BOOKS","DINER","CLUB","LIVE","OPEN"],Ye=["#ff4fb8","#4fe3ff","#ffb547","#9a7bff","#ff5a5a","#5affc8","#ff7ae0","#7ad0ff"];function It(){const i=document.createElement("canvas");i.width=1024,i.height=1024;const t=i.getContext("2d");t.fillStyle="#000",t.fillRect(0,0,1024,1024),Ut.forEach((a,n)=>{const s=n%8*128,r=Math.floor(n/8)*512,e=Ye[n%Ye.length];t.fillStyle="#0b0710",t.fillRect(s+10,r+10,108,492),t.strokeStyle=e,t.lineWidth=5,t.shadowColor=e,t.shadowBlur=16,t.strokeRect(s+16,r+16,96,480),t.fillStyle="#fff",t.font='bold 64px "Oxanium", "Big Shoulders Display", sans-serif',t.textAlign="center",t.textBaseline="middle";const l=a.split(""),c=Math.min(78,440/l.length);l.forEach((p,u)=>{t.shadowBlur=22,t.fillStyle=e,t.fillText(p,s+64,r+256-(l.length-1)*c/2+u*c),t.shadowBlur=0,t.fillStyle="#fff8",t.fillText(p,s+64,r+256-(l.length-1)*c/2+u*c)})});const o=new Me(i);return o.colorSpace=ke,o.anisotropy=4,o}function $t(i,t){if(!t.signs.length)return null;const o=new fe(1.7,6.8),a=new Se;a.index=o.index,a.setAttribute("position",o.attributes.position),a.setAttribute("uv",o.attributes.uv);const n=t.signs.length,s=new Float32Array(n*4),r=new Float32Array(n*2);t.signs.forEach((c,p)=>{s.set([c.x,c.z,c.y,c.cell],p*4),r.set([c.seed,c.side],p*2)}),a.setAttribute("aSign",new oe(s,4)),a.setAttribute("aMeta",new oe(r,2)),a.instanceCount=n;const e=It(),l=new k(a,L({uniforms:{...i,uAtlas:{value:e}},transparent:!0,depthWrite:!1,blending:et,side:ae,vertexShader:`
        attribute vec4 aSign;
        attribute vec2 aMeta;
        varying vec2 vUv;
        varying float vFl, vFog;
        void main() {
          float zc = wrapZ(aSign.y, ${$e}., ${at}.);
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
        }`}));return l.name="ride-signs",l.userData.textures=[e],l}function _t(i){const t=new fe(2400,2400,1,1);t.rotateX(-Math.PI/2),t.translate(0,te+6,-800);const o=new k(t,L({uniforms:i,vertexShader:`
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
        }`}));return o.name="ride-ground",o.renderOrder=-8,o}const I=(i,t,o,a,n)=>i.push(...t,...o,...a,...t,...a,...n);function nt(i,t,o={}){return L({uniforms:i,side:ae,...o,vertexShader:`
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
      }`,fragmentShader:t})}function Xt(i,[t,o]){const a=[],n=[],s=[];for(let x=t;x<o;x+=4){const d=a.length;I(a,[-10.3,0,0],[-10.3,6.4,0],[-10.3,6.4,-4],[-10.3,0,-4]),I(a,[5,0,0],[5,0,-4],[5,6.4,-4],[5,6.4,0]),I(a,[-10.3,6.4,0],[5,6.4,0],[5,6.4,-4],[-10.3,6.4,-4]);for(let f=d;f<a.length;f+=3)n.push(x),s.push(f-d<36?0:1);if(Math.round((x-t)/4)%2===0)for(const f of[-10.3+.25,5-.25]){const v=a.length,w=6.4-.9,m=6.4-.6,h=-.4,y=-2.8,S=f+(f<0?.18:-.18);I(a,[S,w,h],[S,m,h],[S,m,y],[S,w,y]),I(a,[f,w,h],[S,w,h],[S,w,y],[f,w,y]);for(let P=v;P<a.length;P+=3)n.push(x),s.push(2)}}for(const[x,d]of[[t,1],[o,-1]]){const f=a.length,v=0;I(a,[-34,0,v],[-10.3,0,v],[-10.3,24,v],[-34,24,v]),I(a,[5,0,v],[30,0,v],[30,24,v],[5,24,v]),I(a,[-10.3,6.4,v],[5,6.4,v],[5,24,v],[-10.3,24,v]);for(let w=f;w<a.length;w+=3)n.push(x),s.push(3);if(d>0){const w=a.length;I(a,[-6,6.4+1.2,v+.05],[2.5,6.4+1.2,v+.05],[2.5,6.4+3,v+.05],[-6,6.4+3,v+.05]);for(let m=w;m<a.length;m+=3)n.push(x),s.push(4)}}const p=new ce;p.setAttribute("position",new z(a,3)),p.setAttribute("aS",new z(n,1)),p.setAttribute("aK",new z(s,1));const u=new k(p,nt(i,`
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
    }`));return u.name="ride-tunnel",u}function jt(i,[t,o]){const a=[],n=(d,f,v,w,m,h,y,S)=>{const P=new le(f-d,w-v,h-m);P.translate((d+f)/2,(v+w)/2,(m+h)/2),P.deleteAttribute("uv"),P.deleteAttribute("normal");const A=P.attributes.position.count;P.setAttribute("aS",new z(new Array(A).fill(y),1)),P.setAttribute("aK",new z(new Array(A).fill(S),1)),a.push(P)},s=t+170,r=o-190,e=92,l=-24,c=8;for(const d of[s,r]){for(const f of[l,c])n(f-1.6,f+1.6,-46,e,-1.6,1.6,d,0),n(f-.5,f+.5,e,e+1.2,-.5,.5,d,2);for(const f of[28,82])n(l,c,f,f+3,-1.2,1.2,d,0)}const p=d=>{if(d<s)return 2+(e-2)*Math.max(0,(d-t)/(s-t))**1.6;if(d>r)return 2+(e-2)*Math.max(0,(o-d)/(o-r))**1.6;const f=(d-s)/(r-s);return e-(e-10)*4*f*(1-f)};for(const d of[l+1,c-1]){for(let f=t;f<o;f+=6){const v=p(f),w=p(f+6),m=new ce,h=[];I(h,[d-.35,v,0],[d+.35,v,0],[d+.35,w,-6],[d-.35,w,-6]),I(h,[d,v-.35,0],[d,v+.35,0],[d,w+.35,-6],[d,w-.35,-6]),m.setAttribute("position",new z(h,3)),m.setAttribute("aS",new z(new Array(12).fill(f),1)),m.setAttribute("aK",new z(new Array(12).fill(1),1)),a.push(m)}for(let f=t+12;f<o-6;f+=12){const v=p(f);v<3||n(d-.06,d+.06,1,v,-.06,.06,f,1)}}const u=Ie(a.map(d=>d.index?d.toNonIndexed():d)),x=new k(u,nt(i,`
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
    }`));return x.name="ride-bridge",x}const Gt=[["BAY TUNNEL","HARBOUR  2 km"],["BAY BRIDGE","NOCTILUCA BAY"],["EXIT 1 km","MAIN STREET"],["MAIN STREET","KERB  ·  CENTRE"]];function qt(){const i=document.createElement("canvas");i.width=1024,i.height=512;const t=i.getContext("2d");Gt.forEach(([a,n],s)=>{const r=s%2*512,e=Math.floor(s/2)*256;t.fillStyle="#0c5a4a",t.fillRect(r,e,512,256),t.strokeStyle="#e8f4ee",t.lineWidth=8,t.strokeRect(r+14,e+14,484,228),t.fillStyle="#f4faf6",t.textAlign="center",t.font='bold 64px "Oxanium", "Big Shoulders Display", sans-serif',t.fillText(a,r+256,e+110),t.font='44px "Oxanium", "Big Shoulders Display", sans-serif',t.fillText(n,r+256,e+186)});const o=new Me(i);return o.colorSpace=ke,o.anisotropy=4,o}function Wt(i,t){const o=[],a=[],n=[],s=[],r=(u,x,d,f)=>{o.push(...u),a.push(...x),n.push(d),s.push(f)},e=(u,x,d,f,v,w)=>{for(const m of[v,v-.5]){const h=[[u,d,m],[x,d,m],[x,f,m],[u,d,m],[x,f,m],[u,f,m]];for(const y of h)r(y,[0,0],w,0)}};t.forEach(({s:u,board:x})=>{e(-10,-9.6,0,8.2,0,u),e(4.6,5,0,8.2,0,u),e(-10,5,7.6,8.2,0,u);const d=x%2/2,f=1-(Math.floor(x/2)+1)/2,v=-6.8,w=.6,m=4.8,h=7.5,y=.06,S=[[v,m,y,d,f],[w,m,y,d+.5,f],[w,h,y,d+.5,f+.5],[v,m,y,d,f],[w,h,y,d+.5,f+.5],[v,h,y,d,f+.5]];for(const[P,A,Y,D,G]of S)r([P,A,Y],[D,G],u,1)});const l=new ce;l.setAttribute("position",new z(o,3)),l.setAttribute("uv",new z(a,2)),l.setAttribute("aS",new z(n,1)),l.setAttribute("aK",new z(s,1));const c=qt(),p=new k(l,L({uniforms:{...i,uBoards:{value:c}},side:ae,vertexShader:`
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
      }`}));return p.name="ride-gantries",p.userData.textures=[c],p}const st=520,Vt=40;function Yt(){const i=[],t=(o,a,n,s,r,e)=>{const l=new le(o,a,n);l.translate(0,s,r),l.deleteAttribute("uv"),l.setAttribute("kind",new z(new Array(l.attributes.position.count).fill(e),1)),i.push(l)};return t(1.82,.62,4.4,.62,0,0),t(1.5,.5,2.2,1.18,.25,1),Ie(i)}function Qt(){const i=[],t=[],o=[],a=(r,e,l,c,p,u)=>{const x=[[r,l,p],[e,l,p],[e,c,p],[r,l,p],[e,c,p],[r,c,p]],d=[[0,0],[1,0],[1,1],[0,0],[1,1],[0,1]];x.forEach((f,v)=>{i.push(...f),t.push(...d[v]),o.push(u)})},n=(r,e,l,c,p,u)=>{const x=[[r,l,c],[e,l,c],[e,l,p],[r,l,c],[e,l,p],[r,l,p]],d=[[0,0],[1,0],[1,1],[0,0],[1,1],[0,1]];x.forEach((f,v)=>{i.push(...f),t.push(...d[v]),o.push(u)})};a(-1.1,1.1,.35,.95,-2.22,0),a(-1.1,1.1,.5,1,2.22,1);for(const r of[-.7,.7]){const e=[[r,.55,-2.2],[r,.85,-2.2],[r,.85,18],[r,.55,-2.2],[r,.85,18],[r,.55,18]],l=[[0,0],[0,1],[1,1],[0,0],[1,1],[1,0]];e.forEach((c,p)=>{i.push(...c),t.push(...l[p]),o.push(2)})}n(-1.6,1.6,.03,-2.2,-14,3);const s=new ce;return s.setAttribute("position",new z(i,3)),s.setAttribute("uv",new z(t,2)),s.setAttribute("kind",new z(o,1)),s}function Zt(i){const t=[];let o=7;const a=()=>(o=o*16807%2147483647)/2147483647,n=Math.round(i*.4);for(let s=0;s<i;s++){const r=s>=n,e=r?ve.oncoming:[ve.mid,ve.far];t.push({x:e[Math.floor(a()*e.length)]+(a()-.5)*.4,z0:-a()*st,v:r?18+a()*12:16+a()*10,dir:r?1:-1,paint:a()})}return t}const Qe=`
  attribute vec4 aCar; // x, z0, speed, dir (+1 oncoming, -1 our way)
  attribute float aPaint;
  attribute float kind;
  varying vec3 vP;
  varying float vKind, vPaint, vRel, vDir, vS;
  varying vec2 vUv;
  void main() {
    float dir = aCar.w;
    // oncoming cars come toward us at their own speed plus ours; our way, they move ahead of us at theirs
    float z = wrapZ(aCar.y + dir * aCar.z * uT, ${st}., ${Vt}.);
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
  }`;function Ze(i,t){const o=new Se;i.index&&(o.index=i.index);for(const s of Object.keys(i.attributes))o.setAttribute(s,i.attributes[s]);const a=new Float32Array(t.length*4),n=new Float32Array(t.length);return t.forEach((s,r)=>{a.set([s.x,s.z0,s.v,s.dir],r*4),n[r]=s.paint}),o.setAttribute("aCar",new oe(a,4)),o.setAttribute("aPaint",new oe(n,1)),o.instanceCount=t.length,o}function Jt(i,t,{trails:o=!0}={}){const a=new V;a.name="ride-traffic";const n=new k(Ze(Yt(),t),L({uniforms:i,vertexShader:Qe,fragmentShader:`
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
        }`}));n.name="ride-traffic-bodies",a.add(n);const s=Qt(),r=new k(Ze(s,t),L({uniforms:{...i,uTrails:{value:o?1:0}},transparent:!0,depthWrite:!1,blending:et,side:ae,defines:{USE_UV:""},vertexShader:Qe,fragmentShader:`
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
        }`}));return r.name="ride-traffic-lamps",r.renderOrder=3,a.add(r),a}const T={x:.36,z0:-.36,z1:.38,y:1.33};function eo(i){i.updateMatrixWorld(!0);const t=new tt,o=new J,a=new J;i.traverse(n=>{if(!n.isMesh||n.name)return;const s=n.material;s!=null&&s.isMeshBasicMaterial&&s.transparent&&(t.setFromObject(n).getSize(o),t.getCenter(a),o.x<.05&&o.z>2.4&&o.z<2.8&&Math.abs(a.x)>.7&&a.y>1&&a.y<1.4&&(n.name=a.x>0?"glass-right":"glass-left"))})}function to(i,t,o){let a=null,n=1/0;const s=new J;return i.traverse(r=>{if(r.name!==t)return;r.getWorldPosition(s);const e=o?s.distanceTo(o):0;e<n&&(n=e,a=r)}),a}function oo(i,t){const o=[],a=new Map,n=new tt;return i.updateMatrixWorld(!0),i.traverse(s=>{if(!s.isMesh||s.userData.ride||s.userData.noCollide||[].concat(s.material).some(e=>e==null?void 0:e.isShaderMaterial)||(n.setFromObject(s),n.max.y<T.y||n.min.x>T.x||n.max.x<-.36||n.min.z>T.z1||n.max.z<T.z0))return;const r=e=>{if(!e||e.transparent)return e;if(!a.has(e)){const l=e.clone();l.onBeforeCompile=c=>{c.uniforms.uRoofOpen=t,c.vertexShader=c.vertexShader.replace("void main() {",`varying vec3 vRoofW;
void main() {`).replace("#include <project_vertex>",`#include <project_vertex>
  vRoofW = (modelMatrix * vec4(transformed, 1.0)).xyz;`),c.fragmentShader=c.fragmentShader.replace("void main() {",`uniform float uRoofOpen;
varying vec3 vRoofW;
void main() {
  if (uRoofOpen > 0.001 && vRoofW.y > ${T.y.toFixed(3)} && abs(vRoofW.x) < ${T.x.toFixed(3)} && vRoofW.z > ${T.z0.toFixed(3)} && vRoofW.z < ${T.z0.toFixed(3)} + ${(T.z1-T.z0).toFixed(3)} * uRoofOpen) discard;`)},l.customProgramCacheKey=()=>"ride-roof",a.set(e,l),o.push(l)}return a.get(e)};s.material=Array.isArray(s.material)?s.material.map(r):r(s.material)}),o}function ao(){const i=document.createElement("canvas");i.width=512,i.height=224;const t=new Me(i);t.colorSpace=ke;const o=new k(new fe(.31,.13),new pe({map:t,toneMapped:!1}));o.position.set(-.012,.983,-.9115),o.rotation.x=-.08,o.name="ride-screen";let a="";return{mesh:o,draw:s=>{const r=JSON.stringify(s);if(r===a)return;a=r;const e=i.getContext("2d");e.fillStyle="#050b14",e.fillRect(0,0,512,224),e.fillStyle="rgba(120,170,230,0.5)",e.fillRect(12,11,488,1),e.font="11px sans-serif",e.fillStyle="#7fa8d8",e.fillText(s.clock,15,25),e.fillText("MEDIA   ROUTE   CAR",190,25);const l=s.station>0;if(e.fillStyle=l?"#13243a":"#0b1420",e.fillRect(14,38,200,172),e.fillStyle=l?"#ff9ab8":"#4a6a8a",e.font="bold 15px sans-serif",e.fillText(ye[s.station].name,24,62),e.font="12px sans-serif",e.fillStyle="#9cc4ee",l){e.fillText(ye[s.station].track,24,84),e.fillText(ye[s.station].artist,24,102);for(let c=0;c<14;c++){const p=8+(c*37+s.beat*13)%40;e.fillStyle="#4aa8ff",e.fillRect(24+c*13,190-p,8,p)}}else e.fillText("F at the screen: tune in",24,84);e.strokeStyle="#2a4a6a",e.lineWidth=4,e.beginPath(),e.moveTo(250,120),e.lineTo(490,120),e.stroke(),e.strokeStyle="#4aa8ff",e.beginPath(),e.moveTo(250,120),e.lineTo(250+240*s.progress,120),e.stroke(),ot.forEach((c,p)=>{const u=250+240*c.t/$;e.fillStyle=s.chapter>=c.n?"#ff9ab8":"#2a4a6a",e.beginPath(),e.arc(u,120,6,0,Math.PI*2),e.fill(),e.fillStyle="#7fa8d8",e.font="10px sans-serif",e.fillText(String(p+1),u-3,142)}),e.fillStyle="#ffffff",e.beginPath(),e.arc(250+240*s.progress,120,8,0,Math.PI*2),e.fill(),s.done?(e.strokeStyle="#ff9ab8",e.lineWidth=3,e.beginPath(),e.moveTo(252,72),e.lineTo(259,80),e.lineTo(273,62),e.stroke(),e.fillStyle="#ff9ab8",e.font="bold 22px sans-serif",e.fillText(s.title,282,80,216)):(e.fillStyle="#9cc4ee",e.font="bold 16px sans-serif",e.fillText(s.title,250,80,248)),e.fillStyle="#9cc4ee",e.font="14px sans-serif",e.fillText(s.eta,250,178,248),t.needsUpdate=!0},tex:t}}function no(){const i=document.createElement("canvas");i.width=512,i.height=224;const t=new Me(i);t.colorSpace=ke;const o=new k(new fe(.33,.13),new pe({map:t,toneMapped:!1}));o.position.set(-.39,.985,-.9115),o.rotation.x=-.08,o.name="ride-cluster";let a="";return{mesh:o,draw:(s,r)=>{const e=`${Math.round(s)}${r}`;if(e===a)return;a=e;const l=i.getContext("2d");l.setTransform(.5,0,0,.5,0,0),Ct(l,1024,448,Math.round(s),r),t.needsUpdate=!0},tex:t}}function so(){const i=new V;i.name="ride-charm";const t=new k(new Ue(.0012,.0012,.11,4),new pe({color:10127984}));t.position.y=-.055;const o=new mt;o.absarc(0,0,.032,Math.PI*.15,Math.PI*1.85,!1),o.absarc(.014,0,.026,Math.PI*1.75,Math.PI*.25,!0);const a=new k(new gt(o,{depth:.003,bevelEnabled:!1,curveSegments:18}),new Ke({color:15918792,emissive:9069624,emissiveIntensity:.5,roughness:.9,side:ae}));return a.position.set(0,-.14,-.0015),a.rotation.z=.25,i.add(t,a),i.position.set(0,1.262,-.63),i}const io=[{rough:.38,set:{roughness:.74,color:4605003}},{rough:.48,set:{roughness:.64,color:3420986}},{rough:.5,set:{roughness:.7,color:2763055}}];function ro(i){const t=new Map;return i.traverse(o=>{const a=o.material;if(!o.isMesh||!(a!=null&&a.isMeshStandardMaterial)||a.metalness>.05)return;const n=io.find(s=>Math.abs(a.roughness-s.rough)<.015);if(n){if(!t.has(a)){const s=a.clone();s.roughness=n.set.roughness,s.color.setHex(n.set.color),t.set(a,s)}o.material=t.get(a)}}),[...t.values()]}function lo(){const i=new V;i.name="ride-lights";const t=new ut(4868704,2102296,.34),o=new De(16758984,1.3);o.position.set(20,30,-40),o.target.position.set(0,0,-10),o.castShadow=!0,o.shadow.mapSize.set(2048,2048),Object.assign(o.shadow.camera,{left:-30,right:30,top:30,bottom:-30,near:1,far:120}),o.shadow.bias=-4e-4,o.shadow.normalBias=.03;const a=new Oe(16761032,60,16,1.4);a.position.set(1.5,7,-20);const n=new Oe(16730832,30,9,1.6);n.position.set(-7,2,-12);const s=new De(10131104,.52);s.position.set(-.2,1.7,1),s.target.position.set(.1,.85,-1);const r=new De(16042144,0);r.position.set(-1.6,1.55,-.9),r.target.position.set(.5,.95,.2);const e=new Oe(8688832,.1,.8,2);return e.position.set(-.2,.95,-.6),i.add(t,o,o.target,a,n,s,s.target,r,r.target,e),{group:i,hemi:t,key:o,pool:a,mag:n,fill:s,win:r,disp:e}}function co(i,t){const o=new V;o.name="ride-cabin";const a={value:0},n=oo(i,a),s=i.getObjectByName("sunroof");s&&(s.visible=!1);const r=new Ke({color:11845078,roughness:.32,metalness:.35,emissive:1316900}),e=new V;e.name="ride-sunroof";const l=T.x*2,c=T.z1-T.z0,p=(T.z0+T.z1)/2;for(const[b,B,q,ne]of[[l+.06,.03,0,T.z0-.015],[l+.06,.03,0,T.z1+.015],[.03,c,-.36-.015,p],[.03,c,T.x+.015,p]]){const ge=new k(new le(b,.012,B),r);ge.position.set(q,1.392,ne),e.add(ge)}const u=new pe({color:3816026,transparent:!0,opacity:.35,depthWrite:!1,side:ae}),x=new k(new fe(l,c),u);x.rotation.x=Math.PI/2,x.position.set(0,1.455,p),o.add(e,x);const d=so();o.add(d);const f=ao();o.add(f.mesh);const v=no();o.add(v.mesh);const w=new pe({color:3866522,toneMapped:!1}),m=new Ke({color:2764344,roughness:.45,metalness:.5}),h=new V;h.name="ride-lightswitch",h.position.set(-.72,.79,-.858);const y=new k(new Ue(.024,.026,.022,20).rotateX(Math.PI/2),m);y.position.z=.011;const S=new k(new le(.004,.018,.004),w);S.position.set(0,.01,.023),y.add(S);const P=new k(new vt(.005,12),w);P.position.set(.04,.012,.002),h.add(y,P),o.add(h),n.push(r,u,w,m,f.mesh.material,f.tex,v.mesh.material,v.tex);for(const b of[d,e,x,f.mesh,v.mesh,h])b.traverse(B=>{B.userData.ride=!0,B.raycast=()=>{}});o.traverse(b=>{b.userData.noCollide=!0});const A=i.getObjectByName("wheel"),Y=A==null?void 0:A.quaternion.clone();A&&n.push(...ro(A));const D=to(i,"glass-right",new J(1,1.2,0)),G=(D==null?void 0:D.position.y)??0,K=i.getObjectByName("leadcar"),ue=K==null?void 0:K.position.clone(),me=new J(0,0,1),Pe=new pt,Q={t0:-99,amp:0};return{group:o,owned:n,uniforms:{uRoofOpen:a},flick(b){Q.t0=b,Q.amp=.75},screen:f,update(b,B){a.value=b.roof,v.draw(b.speed*3.6,b.t>$-.5?"P":"D"),x.position.z=p+c*b.roof*.98,x.visible=b.roof>.01,D&&(D.position.y=G-.44*b.window),A&&A.quaternion.copy(Y).multiply(Pe.setFromAxisAngle(me,C.clamp(-b.curve*900,-.5,.5)));const q=B-Q.t0,ne=Q.amp*Math.exp(-q/2.2)*Math.sin(q*5.2);d.rotation.x=ne+b.accel*.03+Math.sin(B*1.7)*.015*Math.min(1,b.speed/10),d.rotation.z=C.clamp(b.curve*b.speed*b.speed*.5,-.4,.4)+ne*.2,w.color.setHex(b.head?3866522:1058330),y.rotation.z=b.head?-.9:0,K&&(K.position.x=ue.x-1+Math.sin(B*.31)*.12,K.position.z=ue.z-Math.max(0,b.t-($-14))**2*.9,K.visible=K.position.z>-400)},dispose(){var b;for(const B of n)(b=B.dispose)==null||b.call(B)}}}const Je={high:{traffic:46,rain:1400,signs:!0,midRow:!0,trails:!0},medium:{traffic:34,rain:800,signs:!0,midRow:!0,trails:!0},low:{traffic:24,rain:360,signs:!0,midRow:!1,trails:!1},software:{traffic:14,rain:0,signs:!1,midRow:!1,trails:!1}},be={from:"kouka",title:"Kouka",text:"Saw your car come out of the bay tunnel on the traffic feed. The water is glowing again tonight, the plankton bloom. Look left before the bridge ends."},Le=i=>{try{return new URLSearchParams(location.search).get(i)}catch{return null}};function fo(i){var o,a;const t=i.getObjectByName("leadcar");t&&t.parent!==i&&i.attach(t);for(const n of["road","wet","streaks","lights","skyDome"]){const s=i.getObjectByName(n);s&&(s.visible=!1)}i.traverse(n=>{var s;(s=n.userData)!=null&&s.update&&/mist/i.test(n.name||"")&&n.parent===i&&(n.visible=!1)});for(const n of i.children)n.isMesh&&((a=(o=n.material)==null?void 0:o.uniforms)!=null&&a.uSpeed)&&n!==t&&(n.visible=!1)}function mo(i,{quality:t}={}){const o=ie.tier||(t==="low"?"low":"high"),a=Je[o]||Je.high,n=Le("tod")?Le("tod")!=="day":wt(),s=Le("rain")!=="0",r=i.root;eo(r),fo(r),i.walk={seated:!0,bounds:[-.3,-.3,.3,.3],eye:1.2,floorY:()=>0,colliders:[],start:{position:[0,1.2,.02],target:[0,1.06,-1]},fov:64},i.env=i.env||{},i.env.fog=n?{color:1709104,density:.0026}:{color:6178386,density:.0024},i.env.background=n?395278:3811908;const e=zt();e.uTunnel.value.set(...O.tunnel),e.uBridge.value.set(...O.bridge),e.uRamp.value=O.ramp,e.uEnd.value=O.end,e.uNight.value=n?1:0,e.uFogColor.value.setHex(i.env.fog.color,xt),e.uFogDensity.value=i.env.fog.density;const l=new V;l.name="ride";const c=Lt({midRow:a.midRow,signs:a.signs}),p=[],u=g=>{g&&(l.add(g),p.push(...g.userData.textures||[]))};u(Dt(e)),u(_t(e)),u(Bt(e)),u(Ft(e)),u(Kt(e,c)),u($t(e,c)),u(Xt(e,O.tunnel)),u(jt(e,O.bridge)),u(Wt(e,[{s:420,board:0},{s:O.tunnel[0]-170,board:0},{s:O.bridge[0]+90,board:1},{s:O.bridge[1]-260,board:2},{s:O.ramp+60,board:3}])),u(Jt(e,Zt(a.traffic),{trails:a.trails})),u(At(e,s?a.rain:0)),Ve(l),r.add(l);const x={uniforms:{uWipe:{value:0},uWipeRate:{value:0}}},d=co(r),f=Ve(Tt(e,x)),v=Et();v.group.traverse(g=>{g.raycast=()=>{},g.userData.noCollide=!0});const w=lo();r.add(d.group,f,v.group,w.group);const m=yt(),h={t:0,chapter:1,station:0,roof:0,window:0,wipers:0,head:!0,arrived:!1,rain:0},y={roof:0,window:0},S=new Set,P=new Set,A=g=>{for(const E of P)try{E(g)}catch(F){console.error(F)}};let Y=performance.now(),D=0,G=!1;const K=()=>{Y=performance.now()};document.addEventListener("visibilitychange",K);const ue=(g,E={})=>A({kind:"toast",text:g,...E}),me={"radio:next":()=>{m.arm();const g=m.setStation(m.stationIndex+1===ye.length?0:m.stationIndex+1);return h.station=m.stationIndex,A({kind:"radio",station:h.station}),g.url?`${g.name} · "${g.track}", ${g.artist}`:"The radio clicks off. Just the road now."},"radio:off":()=>(m.arm(),m.setStation(0),h.station=0,A({kind:"radio",station:0}),"The radio clicks off. Just the road now."),"sunroof:toggle":()=>(m.arm(),m.click(),y.roof=y.roof>.5?0:1,y.roof?"The sunroof slides back. Cold air, and the towers leaning in overhead.":"The sunroof slides shut. The city sound drops away."),"window:toggle":()=>(m.arm(),m.click(),y.window=y.window>.5?0:1,y.window?"The window winds down. The wind comes in loud, smelling of rain and the sea.":"The window winds up. Quiet again."),"wipers:cycle":()=>(m.arm(),m.click("wiper"),h.wipers=(h.wipers+1)%3,h.wipers===0&&(G=!0),["Wipers off.","Wipers on.","Wipers fast."][h.wipers]+(h.rain<.05&&h.wipers?" The glass is dry; they squeak once and keep time.":"")),"lights:toggle":()=>(m.arm(),m.click(h.head?"off":"switch"),h.head=!h.head,h.head?"Headlamps on. The lane comes back out of the dark.":"Headlamps off. Between the street lamps the road goes dark; the car does not mind."),"charm:flick":()=>(d.flick(Pe()),"You flick the paper moon. It spins on its thread, and the dashboard light runs around its edge."),"route:show":()=>{const g=We(h.t),E=Math.max(0,Math.round($-h.t));return h.arrived?"Main street, at the kerb. The route screen says: you have arrived.":`Chapter ${g.n} of 3, ${g.title}. Main street in ${Math.floor(E/60)}:${String(E%60).padStart(2,"0")}.`},"ride:restart":()=>(Q.seek(0),S.clear(),"The car pulls away from the kerb again."),horn:()=>(m.arm(),m.click(),null)},Pe=()=>performance.now()/1e3,Q={get state(){return{...h,length:Fe,end:$}},message:be,subscribe(g){return P.add(g),()=>P.delete(g)},use(g){var N,R,X,M;const E=(X=(R=(N=g==null?void 0:g.interactable)==null?void 0:N.primary)==null?void 0:R.effect)==null?void 0:X.device;if(!E||!me[E])return!1;const _=me[E]()||((M=g.lines)==null?void 0:M[0])||null;return _&&ue(_,{device:E}),!0},seek(g){h.t=Math.max(0,Math.min(Fe,g)),Y=performance.now(),g<Be.mid&&S.delete("mid"),g<$&&(S.delete("arrive"),h.arrived=!1)},audio:m};i.ride=Q;const b={pos:new J,rx:0,ry:0,rz:0,on:!1,p0:new J,r0:new bt},B=(g,E,F,_)=>{const N=performance.now(),R=Math.min(1,(N-Y)/1e3);Y=N;const X=document.hidden||!!ie.hold||!!ie.settling;X||(h.t=Math.min(Fe,h.t+R));const M=h.t,U=St(M),j=Ne(M),ee=Mt(M),Ae=(Ne(M+.25)-Ne(M-.25))/.5;e.uT.value=N/1e3%3600,e.uD.value=U,e.uSpeed.value=j,e.uCurve.value=ee,e.uHill.value=kt(M),e.uHeading.value=Pt(M),h.rain=s?C.clamp((M-Be.rain)/10,0,1)*(M>$+2?.8:1):0,e.uRain.value=h.rain,e.uHead.value=C.lerp(e.uHead.value,h.head?1:0,Math.min(1,R*6)),h.roof+=C.clamp(y.roof-h.roof,-R*.7,R*.7),h.window+=C.clamp(y.window-h.window,-R*.6,R*.6);const Te=[0,.75,1.35][h.wipers];if(Te)D+=R*Te,G=!1;else if(G){const Z=Math.ceil(D);D=Math.min(Z,D+R*.75),D>=Z&&(G=!1)}x.uniforms.uWipe.value=D,x.uniforms.uWipeRate.value=Te||(G?.75:0);const Ee=D%1,it=Ee<.5?Ee*2:2-Ee*2;for(const Z of v.blades)Z.rotation.z=it*Rt;d.update({t:M,roof:h.roof,window:h.window,curve:ee,speed:j,accel:Ae,head:h.head},N/1e3);const se=We(M),Re=Math.max(0,Math.round($-M)),we=h.arrived||Re===0;d.screen.draw({station:h.station,beat:h.station?Math.floor(N/500):0,chapter:se.n,done:we,title:we?"Arrived":`${se.n} · ${se.title}`,progress:we?1:Math.min(1,M/$),eta:we?"Main street, at the kerb":`Main street ${Math.floor(Re/60)}:${String(Re%60).padStart(2,"0")}`,clock:new Date().toTimeString().slice(0,5)});const H=U>O.tunnel[0]&&U<O.tunnel[1]?1:0,de=U>O.bridge[0]+40&&U<O.bridge[1]-40?1:0,rt=U%W/W,xe=Math.exp(-(((rt-.5)*W/7)**2)),_e=Math.exp(-(((U%8-4)/2.2)**2)),{key:Xe,hemi:Ce,pool:lt,mag:je,fill:Ge,win:ct}=w;Xe.intensity=H?.6+.9*_e:(de?.95:1.1)+.35*xe,Xe.color.setHex(H?16754768:de?12110079:n?16758984:16765620),Ce.intensity=H?.2+.35*_e:n?de?.42:.3+.22*xe:.5+.1*xe,Ce.color.setHex(H?16752736:n?4999256:8288392),ct.intensity=n||H?0:.75,Ce.groundColor.setHex(de?932410:H?3810320:2102296),lt.intensity=(H?25:60)*(h.head?1:.35);const ft=Math.floor(U/18)%3;je.color.setHex(H?16748608:de?4247776:[16730832,4905215,16742970][ft]),je.intensity=H?10:30;const dt=(n?.52:.85)*(H?.85:ie.still?1:.94+.16*xe);if(Ge.intensity=C.lerp(Ge.intensity,dt,Math.min(1,R*8)),m.update({speed:j,wind:Math.max(h.window,h.roof*.6),rain:h.rain,tunnel:H,paused:X}),_!=null&&_.warp&&ne(_.warp,{v:j,accel:Ae,tun:H,D:U,camera:F}),(se.n!==h.chapter||!S.has("chapter1"))&&(h.chapter=se.n,S.add("chapter1"),A({kind:"chapter",chapter:se})),!S.has("start")&&M>2.5&&(S.add("start"),He("ride:start",{place:"car-1",chapters:ot.length,length:$})),!S.has("mid")&&M>=Be.mid&&(S.add("mid"),He("ride:mid",{place:"car-1",where:"bay-bridge",hie:be.from,message:be.text}),A({kind:"message",message:be})),!S.has("arrive")&&M>=$&&(S.add("arrive"),h.arrived=!0,He("ride:arrive",{place:"car-1",to:"street-1"}),A({kind:"arrive"})),F&&!ie.still){const Z=Math.min(1,j/20);b.pos.set(Math.sin(N*.0021)*.002*Z,(Math.sin(N*.0083)*.0022+Math.sin(N*.0191)*.0011)*Z+(ge(N)-.5)*6e-4*Z,0),b.rz=C.clamp(-ee*j*j*.035,-.035,.035),b.ry=C.clamp(ee*18,-.03,.03),b.rx=C.clamp(-Ae*.004,-.02,.02),b.p0.copy(F.position),b.r0.copy(F.rotation),F.position.add(b.pos),F.rotation.x+=b.rx,F.rotation.y+=b.ry,F.rotation.z+=b.rz,b.on=!0}},q=new J,ne=(g,{v:E,accel:F,tun:_,D:N,camera:R})=>{const X=ie.still||o==="software"||!R,M=C.smoothstep(E,17,25),U=C.clamp(F/3,0,1),j=N-O.tunnel[1];if(g.amount=X?0:Math.min(.85,.5*M+.3*U+.25*_),g.lines=X?0:Math.min(.6,C.smoothstep(F,.3,1)*M+.35*_),g.flash=X||j<0?0:.8*Math.exp(-j/10),g.near=2.5,!X){const ee=C.smoothstep(-R.getWorldDirection(q).z,.35,.75);g.amount*=ee,g.lines*=ee,q.set(0,1.2,-100).project(R),g.center[0]=C.clamp(q.x*.5+.5,0,1),g.center[1]=C.clamp(q.y*.5+.5,0,1)}},ge=g=>{const E=Math.sin(g*12.9898)*43758.5453;return E-Math.floor(E)};i.update=B,i.afterRender=g=>{b.on&&(g.position.copy(b.p0),g.rotation.copy(b.r0),b.on=!1)};const ze=i.dispose;return i.dispose=()=>{ze==null||ze(),document.removeEventListener("visibilitychange",K),m.dispose(),d.dispose(),P.clear(),l.traverse(g=>{g.isMesh&&(g.geometry.dispose(),g.material.dispose())});for(const g of p)g.dispose();f.geometry.dispose(),f.material.dispose(),v.material.dispose()},Q}export{mo as attachRide};
