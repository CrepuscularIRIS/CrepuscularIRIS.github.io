import{G as se,e as P,O as zt,U as Ve,a0 as at,Z as de,$ as T,a4 as Ne,a5 as ae,P as be,X as le,L as ye,b7 as kt,S as Ae,aa as Pt,aj as At,af as Tt,aG as Et,C as Be,y as ft,d as dt,F as re,a7 as We,M as Pe,ac as Rt,bG as H,ao as Ft,bH as E,R as Ct,T as Ge,Q as Fe,a2 as Dt,ak as Lt,bI as Le,bJ as ht,c as me,bK as Ot,bL as N,bM as Nt,as as Bt,bN as qe,bO as je,bP as Kt,bQ as It,bR as He,bS as Ut,bT as $t,bU as _t,bV as ot,bW as Xe}from"./index-CAG_hBLn.js";import{ax as $,ay as Gt,az as qt,aA as st,aB as jt,aC as Ht,aD as Xt}from"./texlocal-h1DL1jsx.js";import{paintCluster as Wt}from"./ctex-ByXsZQiB.js";import"./react-three-fiber.esm-CyB8prmV.js";const nt=(7*Math.PI/6+.45)/(2*Math.PI);function Vt(a){const e=new se;e.name="ride-sky";const s=new P(new zt(900,32,16),$({uniforms:a,side:at,depthWrite:!1,vertexShader:`
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
          // (dusk: the afterglow keeps to a low band; above it a dusky violet, not a sky-wide pink)
          vec3 mid = mix(vec3(.5, .37, .5), vec3(.10, .06, .17), uNight);
          vec3 hor = mix(vec3(1., .55, .32), vec3(.55, .2, .38), uNight);
          vec3 col = mix(hor, mid, smoothstep(0., mix(.1, .16, uNight), h));
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
          // clouds lit from below by the city: thin bands drifting (value noise: the raw cells were 24 x 6 degree
          // rectangles, read as pink panels hung in the sky, twice as bright in the rain)
          vec2 cq = vec2(g.x * .02 + uT * .01, g.y * .08), ci = floor(cq), cf = fract(cq);
          cf = cf * cf * (3. - 2. * cf);
          float cn = mix(mix(hash12(ci), hash12(ci + vec2(1., 0.)), cf.x), mix(hash12(ci + vec2(0., 1.)), hash12(ci + vec2(1., 1.)), cf.x), cf.y);
          float cl = smoothstep(.5, .85, cn) * smoothstep(.0, .1, h) * (1. - smoothstep(.25, .5, h));
          col += vec3(.35, .12, .28) * cl * .25 * (1. + uRain);
          gl_FragColor = lin(col);
        }`}));s.renderOrder=-10,e.add(s);const o=new P(new Ve(650,650,260,96,1,!0),$({uniforms:a,side:at,transparent:!0,depthWrite:!1,vertexShader:`
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
          // the downtown peak ahead-left over the bridge (where Kouka says to look), with open sky either side of it:
          // neon-zenith layout.js's far-shore ridge (MIT), h += ridge^2 * range, on the ring's periodic angle
          float d1 = abs(fract((wcol + .5) / 100. - ${nt.toFixed(4)} + .5) - .5);
          float d2 = abs(fract((floor(u / 6.) + .5) / 150. - ${nt.toFixed(4)} + .5) - .5);
          float r1 = max(0., 1. - d1 / .045), r2 = max(0., 1. - d2 / .05);
          float open1 = smoothstep(.045, .065, d1) * (1. - smoothstep(.08, .14, d1));
          float open2 = smoothstep(.05, .07, d2) * (1. - smoothstep(.085, .14, d2));
          h1 = h1 * (1. - .6 * open1) + r1 * r1 * 120. * (.55 + .45 * hash11(wcol * 5.3));
          h2 = h2 * (1. - .6 * open2) + r2 * r2 * 40. * (.5 + .5 * hash11(floor(u / 6.) * 3.7));
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
        }`}));return o.renderOrder=-9,e.add(o),e}const ke={mid:-3.6,far:-7.2,oncoming:[-12.2,-15.8,-19.2],parapet:4.6,median:-9.8},oe=32,xe=760,ze=14;function Yt(a,e,s=190){const o=new be(e-a,xe,1,s);return o.rotateX(-Math.PI/2),o.translate((a+e)/2,0,ze-xe/2),o}function Zt(a){const e=new se;e.name="ride-road";const s=new P(Yt(-22,5.2),$({uniforms:a,vertexShader:`
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
          float lp = mod(s, ${oe}.), lq = mod(s + 16., ${oe}.);
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
        }`}));s.name="ride-road-deck",e.add(s);const o=new de,i=[],n=[],r=(d,v,w,g)=>{for(let c=0;c<190;c++){const h=ze-xe*c/190,p=ze-xe*(c+1)/190;i.push(d,v,h,d,w,h,d,v,p,d,w,h,d,w,p,d,v,p);for(let m=0;m<6;m++)n.push(g)}},t=(d,v,w,g)=>{for(let c=0;c<190;c++){const h=ze-xe*c/190,p=ze-xe*(c+1)/190;i.push(d,w,h,v,w,h,d,w,p,v,w,h,v,w,p,d,w,p);for(let m=0;m<6;m++)n.push(g)}};r(-9.4,0,.85,0),r(-10.2,0,.85,0),t(-10.2,-9.4,.85,1),r(4.4,0,1.15,2),t(4.4,4.9,1.15,3),r(4.9,-1.2,1.15,2),r(-22,-1.2,.9,2),r(5.2,-1.4,0,4),o.setAttribute("position",new T(i,3)),o.setAttribute("kind",new T(n,1));const l=new P(o,$({uniforms:a,side:le,vertexShader:`
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
        }`}));return l.name="ride-barriers",e.add(l),e}function Qt(a){const e=[],s=(d,v,w,g,f,c,h)=>{const p=new ye(d,v,w);p.translate(g,f,c),p.setAttribute("glow",new T(new Array(p.attributes.position.count).fill(h),1)),p.deleteAttribute("uv"),e.push(p)};s(.22,10,.22,0,5,0,0),s(2.8,.14,.14,-1.4,9.8,0,0),s(.9,.12,.38,-2.6,9.68,0,1);const o=Ye(e),i=Math.ceil(760/oe),n=i*2,r=new Ne;r.index=o.index;for(const d of Object.keys(o.attributes))r.setAttribute(d,o.attributes[d]);const t=new Float32Array(n*3);for(let d=0;d<i;d++)t.set([ke.parapet+.2,-d*oe+2,1],d*3),t.set([ke.median,-d*oe-14,-1],(i+d)*3);r.setAttribute("slot",new ae(t,3)),r.instanceCount=n;const l=new P(r,$({uniforms:a,vertexShader:`
        attribute float glow;
        attribute vec3 slot;
        varying float vGlow, vS;
        varying vec3 vP;
        void main() {
          vec3 p = position;
          p.x *= slot.z;
          float z = wrapZ(slot.y, ${i*oe}., 14.);
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
        }`}));return l.name="ride-lamps",l}function Ye(a){const e=Object.keys(a[0].attributes),s=new de;let o=0;const i=[],n=Object.fromEntries(e.map(r=>[r,[]]));for(const r of a){const t=r;for(const l of e)n[l].push(...t.attributes[l].array);if(t.index)for(const l of t.index.array)i.push(l+o);else for(let l=0;l<t.attributes.position.count;l++)i.push(l+o);o+=t.attributes.position.count}for(const r of e)s.setAttribute(r,new T(n[r],a[0].attributes[r].itemSize));return s.setIndex(i),s}const we={GLASS:1,OFFICE:3};function Jt(a){let e=a>>>0;return function(){e=e+1831565813>>>0;let o=e;return o=Math.imul(o^o>>>15,o|1),o^=o+Math.imul(o^o>>>7,o|61),((o^o>>>14)>>>0)/4294967296}}const b=(a,e,s)=>e+(s-e)*a();function ea(a,e){for(let s=e.length-1;s>0;s--){const o=Math.floor(a()*(s+1));[e[s],e[o]]=[e[o],e[s]]}return e}const ta=(a,e=0)=>Jt(Math.floor(a*2**31)^e*2654435761),X=(a,e,s,o,i,n,r={})=>({x:a,z:e,w:s,d:o,y0:i,y1:n,...r}),F=(a,e,s,o,i,n,r,t={})=>({x:a,z:e,w:s,d:o,y0:i,y1:n,kind:r,feat:!0,...t});function Oe(a,e=.6,s=.7){return F(a.x,a.z,a.w+e*2,a.d+e*2,a.y1-s,a.y1,"clad")}function ge(a,e,s,o){const i=a(),n=e.y1;if(i<.3){const r=b(a,9,o?26:16),t=Math.min(1.6,e.w*.08);for(const[l,d]of[[-1,-1],[1,-1],[-1,1],[1,1]])s.push(F(e.x+l*(e.w/2-t/2),e.z+d*(e.d/2-t/2),t,t,n,n+r,"clad"));s.push(F(e.x,e.z-e.d/2+t/2,e.w,t,n+r-1.4,n+r,"led")),s.push(F(e.x,e.z+e.d/2-t/2,e.w,t,n+r-1.4,n+r,"led")),s.push(F(e.x-e.w/2+t/2,e.z,t,e.d-2*t,n+r-1.4,n+r,"led")),s.push(F(e.x+e.w/2-t/2,e.z,t,e.d-2*t,n+r-1.4,n+r,"led")),s.push(F(e.x,e.z,e.w*.45,e.d*.45,n,n+b(a,4,7),"plant"))}else if(i<.55){const r=b(a,6,9);s.push(F(e.x,e.z,e.w-1.2,e.d-1.2,n,n+r,"louvre"));const t=b(a,.62,.8);s.push(F(e.x,e.z,e.w*t,e.d*t,n+r,n+r+b(a,8,o?22:12),"led"))}else if(i<.8){let r=e.w,t=e.d,l=n;const d=2+Math.floor(a()*2);for(let v=0;v<d;v++){r*=b(a,.68,.82),t*=b(a,.68,.82);const w=b(a,5,10);s.push(F(e.x,e.z,r,t,l,l+w,v===d-1?"led":"louvre")),l+=w}}else s.push(Oe(e,.5,.9)),s.push(F(e.x+b(a,-.15,.15)*e.w,e.z+b(a,-.15,.15)*e.d,e.w*b(a,.3,.5),e.d*b(a,.3,.5),n,n+b(a,4,8),"plant"));o&&a()<.75&&s.push(F(e.x,e.z,1.6,1.6,n,n+b(a,24,70),"clad",{mast:!0}))}function aa(a,e,s,o){const{x:i,z:n,w:r,d:t}=e,l=[X(i,n,r,t,0,o)],d=a(),v=s>300,w=Math.max(r,t)>46,g=r>=t;if(d<.24||d<.4&&!w){const f=Math.min(r,t)*b(a,.55,.8);let c=Math.min(r-4,f*b(a,.9,1.3)),h=Math.min(t-4,f),p=o;const m=s>400?3:s>200?2:1,u=b(a,-.12,.12)*(r-c),y=b(a,-.12,.12)*(t-h);for(let z=0;z<m;z++){const k=z===m-1?s:p+(s-p)*b(a,.45,.65),M=X(i+u,n+y,c,h,p,k);l.push(M),z<m-1?(a()<.6&&(M.y1=k-b(a,4,7),l.push(F(M.x,M.z,c-.8,h-.8,M.y1,k,"louvre"))),l.push(Oe({...M,y1:k},.8,.9))):ge(a,M,l,v),p=k,c*=b(a,.7,.86),h*=b(a,.7,.86)}}else if(d<.5){const f=s>380?5:s>200?4:3;let c=o;const h=s-o;for(let p=0;p<f;p++){const m=p===f-1?s-c:h/f*b(a,.8,1.2),u=Math.min(r-2,r*b(a,.5,.82)),y=Math.min(t-2,t*b(a,.5,.82)),z=b(a,-1,1)*Math.max(0,(r-u)/2-1),k=b(a,-1,1)*Math.max(0,(t-y)/2-1),M=X(i+z,n+k,u,y,c,c+m,{style:p%2?we.OFFICE:we.GLASS});if(l.push(M),p<f-1&&a()<.55){const I=b(a,4.5,8);M.y1-=I,l.push(F(M.x,M.z,u*.55,y*.55,M.y1,M.y1+I,a()<.5?"louvre":void 0,{style:we.GLASS}))}l.push(Oe(M,.4,.8)),p===f-1&&ge(a,M,l,v),c+=m}}else if(d<.7&&w){const f=g?r:t,c=g?t:r,h=Math.max(9,f*b(a,.14,.22)),p=(f-h)/2-2,m=Math.min(c-4,p*b(a,.8,1.2)),u=s*b(a,.78,.95),y=V=>g?[i+V*(h/2+p/2),n]:[i,n+V*(h/2+p/2)],[z,k]=y(-1),[M,I]=y(1),R=X(z,k,g?p:m,g?m:p,o,s),_=X(M,I,g?m:p,g?m:p,o,u,{style:we.GLASS});l.push(R,_);const U=1+Math.floor(a()*3);for(let V=0;V<U;V++){const te=o+(u-o)*b(a,.3,.85),ne=b(a,7,11),Y=b(a,5,8);l.push(F(i,n,g?h+2:ne,g?ne:h+2,te,te+Y,void 0,{style:we.GLASS})),l.push(F(i,n,g?h+2.4:ne+.8,g?ne+.8:h+2.4,te-.8,te,"clad"))}ge(a,R,l,v),ge(a,_,l,v&&u>300)}else if(d<.85){const f=g?r-6:t-6,c=Math.min((g?t:r)-6,b(a,18,28)),h=X(i,n,g?f:c,g?c:f,o,s);l.push(h);const p=b(a,3.6,6.2),m=b(a,.7,1.4),u=Math.floor(f/p);for(let y=0;y<=u;y++){const z=-f/2+(f-u*p)/2+y*p;for(const k of[-1,1])g?l.push(F(i+z,n+k*(c/2+m/2),.45,m,o+2,s-1.5,"clad")):l.push(F(i+k*(c/2+m/2),n+z,m,.45,o+2,s-1.5,"clad"))}ge(a,h,l,v)}else{const f=Math.min(r,t)*b(a,.42,.55),c=X(i,n,f,f,o,s);l.push(c);const h=ea(a,[[1,0],[-1,0],[0,1],[0,-1]]).slice(0,2+Math.floor(a()*3));for(const[p,m]of h){const u=(p?r:t)/2-2,y=u-f/2;if(y<6)continue;const z=f*b(a,.55,.8),k=o+(s-o)*b(a,.3,.7),M=X(i+p*(f/2+y/2),n+m*(f/2+y/2),p?y:z,m?y:z,o,k,{feat:!0,style:we.OFFICE});l.push(M,Oe(M,.4,.8))}ge(a,c,l,v)}return l}function oa(a,e,s,o,i,n,r){const t=[X(e,s,o,i,0,n)],l=a();return l<.35&&n>60?(t[0].y1=n*b(a,.55,.75),t.push(X(e+b(a,-.15,.15)*o,s+b(a,-.15,.15)*i,o*b(a,.55,.8),i*b(a,.55,.8),t[0].y1,n))):l<.5&&n>80&&t.push(X(e,s,o*.7,i*.7,n,n+b(a,8,30),{feat:!0,kind:"led"})),t}function sa(a,e=-46){const s=a.h-e,o=ta(a.seed,71+a.row),n=(a.row>=2?oa(o,0,0,a.w,a.d,s):aa(o,{x:0,z:0,w:a.w,d:a.d},s,Math.min(12,s*.12))).filter(v=>[v.x,v.z,v.w,v.d,v.y0,v.y1].every(Number.isFinite)&&v.w>.05&&v.d>.05&&v.y1-v.y0>.05),r=n.filter(v=>!v.feat),t=n.filter(v=>v.feat),l=a.row>=2?2:a.row===1?10:12,d=t.length<=l?t:[...t.slice(0,Math.floor(l/2)),...t.slice(-Math.ceil(l/2))];return[...r,...d]}function na(a,e=-46){return a.flatMap(s=>sa(s,e).map(o=>({parent:s,block:o})))}const Ze=720,vt=40,W=-46,ee=170,ia=150,ra=.6,fe=5;function la(a){let e=a>>>0;return()=>{e=e+1831565813>>>0;let s=e;return s=Math.imul(s^s>>>15,s|1),s^=s+Math.imul(s^s>>>7,s|61),((s^s>>>14)>>>0)/4294967296}}const ca=[{face:9.5,side:1,w:[9,18],d:[10,22],h:[22,95],row:0},{face:34,side:1,w:[12,26],d:[14,30],h:[45,170],row:1},{face:-25,side:-1,w:[9,18],d:[10,22],h:[22,95],row:0},{face:-52,side:-1,w:[12,26],d:[14,30],h:[45,170],row:1},{face:92,side:1,w:[20,34],d:[18,34],h:[80,205],row:2},{face:-112,side:-1,w:[20,34],d:[18,34],h:[80,205],row:2}];function fa(a){const e=la(20261020),s=[],o=[];for(const i of ca){if(i.row===1&&!a.midRow||i.row===2&&!a.farRow)continue;let n=0;for(;n<Ze-8;){const r=i.w[0]+e()*(i.w[1]-i.w[0]),t=i.d[0]+e()*(i.d[1]-i.d[0]),l=i.h[0]+Math.pow(e(),1.6)*(i.h[1]-i.h[0]),d=i.row===2?10+e()*14:1.5+e()*(i.row?10:5),v=i.face+i.side*(t/2+e()*(i.row?14:3)),w=-(n+r/2);if(s.push({x:v,z:w,w:t,d:r,h:l,seed:e(),row:i.row,side:i.side}),i.row===0&&a.signs&&e()<.62){const g=1+(e()<.3?1:0);for(let f=0;f<g;f++){const c=v-i.side*(t/2)-i.side*1.1;o.push({x:c,z:w+r/2-.5-f*(r-1),y:7+e()*14,cell:Math.floor(e()*16),seed:e(),side:i.side})}}n+=r+d}}return{towers:s,signs:o}}function da(a){const e={value:0};if(!a)return{a:null,m:null,ready:e,textures:[]};const s=new kt;let o=2;const i=()=>{--o===0&&(e.value=1)},n=l=>{o=-1,console.warn("ride facade texture failed; procedural fallback retained",l)},r=s.load(new URL("./places/ride/facades_albedo.webp",document.baseURI).href,i,void 0,n),t=s.load(new URL("./places/ride/facades_mask.webp",document.baseURI).href,i,void 0,n);for(const[l,d]of[[r,Ae],[t,Pt]])l.colorSpace=d,l.wrapS=l.wrapT=At,l.minFilter=Tt,l.magFilter=Et,l.anisotropy=4,l.flipY=!1,l.generateMipmaps=!0;return{a:r,m:t,ready:e,textures:[r,t]}}function ha(a,e,{facades:s=!0}={}){const o=new ye(1,1,1);o.translate(0,.5,0);const i=new Ne;i.index=o.index,i.setAttribute("position",o.attributes.position),i.setAttribute("normal",o.attributes.normal);const n=na(e.towers,W),r=da(s),t=n.length,l=new Float32Array(t*4),d=new Float32Array(t*4),v=new Float32Array(t*4),w=new Float32Array(t*4),g={clad:1,louvre:2,led:3,plant:4};n.forEach(({parent:c,block:h},p)=>{l.set([c.x,c.z,c.w,c.d],p*4),d.set([c.h,c.seed,c.row,c.side],p*4),v.set([h.x,h.z,h.y0,h.y1],p*4),w.set([h.w,h.d,g[h.kind]||0,h.feat?1:0],p*4)}),i.setAttribute("aBox",new ae(l,4)),i.setAttribute("aMeta",new ae(d,4)),i.setAttribute("aMass",new ae(v,4)),i.setAttribute("aSize",new ae(w,4)),i.instanceCount=t;const f=new P(i,$({uniforms:{...a,uFacadeA:{value:r.a},uFacadeM:{value:r.m},uFacadeReady:r.ready},vertexShader:`
        attribute vec4 aBox;
        attribute vec4 aMeta;
        attribute vec4 aMass;
        attribute vec4 aSize;
        varying vec3 vP, vN, vL;
        varying float vSeed, vH, vS, vRow, vKind, vTop, vFeat, vShore;
        void main() {
          float h = aMeta.x, seed = aMeta.y, row = aMeta.z, side = aMeta.w;
          float zc = wrapZ(aBox.y, ${Ze}., ${vt}.);
          float s = uD - zc;
          float bri = onBridge(s);
          float ms = mainStreet(s);
          // the far-shore district: right-hand back rows step out and sink only partway (their feet under the shore)
          float shore = step(.5, row) * step(0., side) * step(hash11(seed * 61.), ${ra});
          float x = aBox.x - side * ms * (row < .5 ? 2.5 : 10.) + shore * bri * ${ia}.;
          float hh = h * (1. - .25 * ms * max(0., 1. - row));
          float hs = (hh - ${W}.) / (h - ${W}.);
          float bh = aMass.w - aMass.z;
          vec3 p = position * vec3(aSize.x, bh * hs, aSize.y);
          p.y += ${W}. + aMass.z * hs;
          p.y -= bri * mix(hh + 120., hh * .5 + 10., shore); // the terrace (buildShore) hides the feet
          vShore = shore * bri;
          p.x += x + aMass.x;
          p.z += zc + aMass.y;
          vL = vec3(aMass.x + position.x * aSize.x, ${W}. + (aMass.z + position.y * bh) * hs, aMass.y + position.z * aSize.y);
          vN = normal;
          vSeed = floor(seed * 4096.) / 4096.;
          vH = hh;
          vTop = ${W}. + aMass.w * hs;
          vS = s;
          vRow = row;
          vKind = aSize.z;
          vFeat = aSize.w;
          vP = p;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(bend(p), 1.);
        }`,fragmentShader:`
        uniform sampler2D uFacadeA;
        uniform sampler2D uFacadeM;
        uniform float uFacadeReady;
        varying vec3 vP, vN, vL;
        varying float vSeed, vH, vS, vRow, vKind, vTop, vFeat, vShore;
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
          // neon-zenith feature blocks: cladding, mechanical louvres, lit crowns and roof plant stay window-free.
          if (vKind > .5) {
            vec3 c = base * .58;
            if (vKind < 1.5) c = base * .52 + vec3(.025, .03, .04) * sun;
            else if (vKind < 2.5) {
              float fw = clamp(1.2 - 2. * fwidth(v / 1.2), 0., 1.);
              float slat = mix(.25, smoothstep(.24, .08, abs(fract(v / 1.2) - .5)), fw);
              c = base * (.32 + .42 * slat);
            } else if (vKind < 3.5) {
              float panel = step(.12, fract(u / 2.4)) * step(.18, fract(v / 1.6));
              vec3 cc = mix(vec3(1., .25, .65), vec3(.25, .8, 1.), step(.5, hash11(SEED * 17.)));
              c = base * .25 + cc * (.12 + .48 * panel) * mix(.4, 1.1, uNight);
            } else {
              c = base * .82 + vec3(.035, .04, .045) * sun;
              c += top * step(.78, hash11(SEED * 53.)) * vec3(1., .08, .03);
            }
            float d = length(vP);
            c = mix(c, uFogColor, fogOf(d));
            float air = (1. - uNight) * (1. - exp(-d / 240.)) * (.5 + .25 * vRow);
            c = mix(c, vec3(.66, .53, .5), air);
            gl_FragColor = lin(c);
            return;
          }
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
          // rooms vary a little in brightness (the cubed hash made a few blinding squares among many dim ones), and a
          // lit room is brightest under its ceiling tubes and falls off toward the sill (neon-zenith facadekit.js roomLit's
          // lamp fall-off, MIT, as a gradient on the pane only: no parallax room)
          wc *= mix(.45, .5 + .45 * hash12(cell + 3.), aa);
          float room = mix(1., .6 + .55 * smoothstep(.3, .78, f.y), aa);
          vec3 c = base * (1. + pier * .35);
          // by day a lit office is a faint warm square against glass that holds the sky
          c += wc * room * pane * lit * step(0., v) * 1.15 * mix(1., .48, min(vRow, 1.)) * mix(.45, 1., uNight);
          c += mix(vec3(.17, .17, .22) * (.55 + .6 * sun), vec3(.025, .03, .045), uNight) * pane * (1. - lit);
          // Only near-row, road-facing main-body sides use the licensed atlas. Other rows/low tiers keep the cheap path.
          float usePhoto = uFacadeReady * (1. - step(.5, vRow)) * side * (1. - top) * (1. - vFeat);
          if (usePhoto > .5) {
            float layer = floor(hash11(SEED * 5.) * 4.);
            vec2 dim = vec2(14.834, 14.5), cellId = vec2(0., 0.);
            if (layer > .5 && layer < 1.5) { dim = vec2(14.859, 15.); cellId = vec2(1., 0.); }
            else if (layer > 1.5 && layer < 2.5) { dim = vec2(21.692, 15.5); cellId = vec2(0., 1.); }
            else if (layer > 2.5) { dim = vec2(30.4); cellId = vec2(1., 1.); }
            vec2 tileUv = vec2(fract(u / dim.x), 1. - fract(v / dim.y));
            vec2 inner = (8. + tileUv * 496.) / 512.;
            vec2 auv = (cellId + inner) * .5;
            // SRGB textures arrive linear in WebGL; return to this shader's sRGB working space, then lin() exactly once.
            vec3 pa = pow(max(texture2D(uFacadeA, auv).rgb, vec3(0.)), vec3(1. / 2.2));
            float win = texture2D(uFacadeM, auv).r;
            vec2 pfw = fwidth(vec2(u / dim.x, v / dim.y));
            float paa = clamp(1.5 - 2.4 * max(pfw.x, pfw.y), 0., 1.);
            pa = mix(vec3(.2, .2, .21), pa, paa);
            win = mix(.16, win, paa);
            float rooms = step(1. - ratio * (.75 + .55 * floorOn), hash12(vec2(floor(u / dim.x * 3.), floor(v / dim.y * 5.)) + SEED * 17.));
            float pl = mix(ratio, rooms, paa);
            vec3 photo = pa * mix(.62, .22, uNight) * (.75 + .25 * sun);
            photo += wc * win * pl * mix(.35, .8, uNight);
            photo += vec3(.025, .035, .05) * win * (1. - pl);
            c = photo;
          }
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
          float edge = smoothstep(vTop - .6, vTop - .1, v) * (1. - top) * step(.5, hash11(SEED * 3.));
          c += mix(vec3(1., .3, .6), vec3(.3, .8, 1.), step(.5, hash11(SEED * 17.))) * edge * 1.4;
          c = mix(c, base * .8, top);
          // city glow from below, haze with distance
          c += vec3(.22, .08, .18) * exp(-(v - ${W}.) / 40.) * .6;
          // on the far-shore terrace: the street at the tower's foot lights its lowest floors (shopfronts, lamps)
          c += vec3(1., .62, .38) * .32 * exp(-max(vP.y - ${fe}., 0.) / 7.) * vShore * mix(.4, 1., uNight);
          float d = length(vP);
          c = mix(c, uFogColor, fogOf(d) * (1. - lit * pane * .15));
          // dusk air: the farther towers (and the back row) fade into a lighter haze, the near ones keep their shade
          float air = (1. - uNight) * (1. - exp(-d / 240.)) * (.5 + .25 * vRow);
          c = mix(c, vec3(.66, .53, .5), air);
          gl_FragColor = lin(c);
        }`}));return f.name="ride-towers",f.userData.textures=r.textures,f.userData.stats={parentTowers:e.towers.length,instances:t,draws:1,triangles:t*12},f}const va=["HOTEL","BAR","24H","KTV","NOODLE","CAFE","ARCADE","NOCTILUCA","JINRUI","GOODBOAT","TAXI","BOOKS","DINER","CLUB","LIVE","OPEN"],it=["#ff4fb8","#4fe3ff","#ffb547","#9a7bff","#ff5a5a","#5affc8","#ff7ae0","#7ad0ff"];function pa(){const a=document.createElement("canvas");a.width=1024,a.height=1024;const e=a.getContext("2d");e.fillStyle="#000",e.fillRect(0,0,1024,1024),va.forEach((o,i)=>{const n=i%8*128,r=Math.floor(i/8)*512,t=it[i%it.length];e.fillStyle="#0b0710",e.fillRect(n+10,r+10,108,492),e.strokeStyle=t,e.lineWidth=5,e.shadowColor=t,e.shadowBlur=16,e.strokeRect(n+16,r+16,96,480),e.fillStyle="#fff",e.font='bold 64px "Oxanium", "Big Shoulders Display", sans-serif',e.textAlign="center",e.textBaseline="middle";const l=o.split(""),d=Math.min(78,440/l.length);l.forEach((v,w)=>{e.shadowBlur=22,e.fillStyle=t,e.fillText(v,n+64,r+256-(l.length-1)*d/2+w*d),e.shadowBlur=0,e.fillStyle="#fff8",e.fillText(v,n+64,r+256-(l.length-1)*d/2+w*d)})});const s=new Be(a);return s.colorSpace=Ae,s.anisotropy=4,s}function ua(a,e){if(!e.signs.length)return null;const s=new be(1.7,6.8),o=new Ne;o.index=s.index,o.setAttribute("position",s.attributes.position),o.setAttribute("uv",s.attributes.uv);const i=e.signs.length,n=new Float32Array(i*4),r=new Float32Array(i*2);e.signs.forEach((d,v)=>{n.set([d.x,d.z,d.y,d.cell],v*4),r.set([d.seed,d.side],v*2)}),o.setAttribute("aSign",new ae(n,4)),o.setAttribute("aMeta",new ae(r,2)),o.instanceCount=i;const t=pa(),l=new P(o,$({uniforms:{...a,uAtlas:{value:t}},transparent:!0,depthWrite:!1,blending:ft,side:le,vertexShader:`
        attribute vec4 aSign;
        attribute vec2 aMeta;
        varying vec2 vUv;
        varying float vFl, vFog;
        void main() {
          float zc = wrapZ(aSign.y, ${Ze}., ${vt}.);
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
        }`}));return l.name="ride-signs",l.userData.textures=[t],l}function ma(a){const e=new be(2400,2400,1,1);e.rotateX(-Math.PI/2),e.translate(0,W+6,-800);const s=new P(e,$({uniforms:a,vertexShader:`
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
          // the far shore on the right (the towers kept there stand on it): the lower city's grid behind a lit quay.
          // The towers ahead are bent (glsl bend()), the plane is not: the quay follows the same bend.
          float ahead = max(-vP.z, 0.);
          float xs = x - uCurve * ahead * ahead;
          float land = smoothstep(${ee-1}., ${ee+1}., xs);
          float quay = exp(-pow((xs - ${ee+2}.) / 1.4, 2.)) * (.35 + .65 * smoothstep(.7, .95, fract(s / 11.)));
          sea = mix(sea, city * 1.3 + vec3(1., .8, .58) * quay * .9, land);
          vec3 c = mix(city, sea, bri);
          float d = length(vP - vec3(0., ${W}., 0.));
          c = mix(c, uFogColor, fogOf(d) * .85);
          gl_FragColor = lin(c);
        }`}));return s.name="ride-ground",s.renderOrder=-8,s}function wa(a){const e=[],s=[],r=fe+1.1,t=(v,w,g)=>{for(let f=60;f>-1100;f-=10){const c=f-10,h=[[...v,f],[...w,f],[...w,c],[...v,f],[...w,c],[...v,c]];for(const p of h)e.push(p[0],p[1],p[2]),s.push(g)}};t([ee,W+4],[ee,r],0),t([ee,r],[ee+1.2,r],0),t([ee+1.2,fe],[ee+700,fe],1);const l=new de;l.setAttribute("position",new T(e,3)),l.setAttribute("aF",new T(s,1));const d=new P(l,$({uniforms:a,side:le,vertexShader:`
        attribute float aF;
        varying vec3 vP;
        varying float vF;
        void main() {
          vec3 p = position;
          p.y -= (1. - onBridge(uD - p.z)) * 120.;
          vP = p; vF = aF;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(bend(p), 1.);
        }`,fragmentShader:`
        varying vec3 vP;
        varying float vF;
        void main() {
          float s = uD - vP.z;
          vec3 c;
          if (vF < .5) {
            // dark stone in courses; a lamp every 11 m on the parapet (a few out), each throwing a cone down the wall
            float y = vP.y;
            float course = smoothstep(.02, .1, abs(fract(y / 1.6) - .5));
            c = mix(vec3(.15, .12, .13), vec3(.075, .066, .085), uNight) * (.7 + .3 * course); // dusk: the stone holds the sky
            float i = floor(s / 11.), f = fract(s / 11.) - .5;
            float on = step(.18, hash11(i * 7.13));
            float dz = f * 11.;
            float head = exp(-pow(dz / .45, 2.) - pow((y - ${fe+.7}) / .3, 2.));
            float below = ${fe+.5} - y;
            float cone = exp(-pow(dz / (.8 + below * .35), 2.)) * exp(-max(below, 0.) / 9.) * step(0., below);
            c += on * (vec3(1., .82, .58) * head * 3. + vec3(1., .66, .4) * cone * .9) * mix(.35, 1., uNight);
            // where the wall stands in the bay: the plankton glow laps along its foot
            float lap = .6 + .4 * sin(s * .13 + uT * .7) * sin(s * .041 - uT * .3);
            c += vec3(.1, .95, .85) * .35 * lap * exp(-max(y - ${W+6}., 0.) / 1.5);
            // the parapet itself takes the lamps' light along its face
            c += vec3(.5, .38, .28) * .25 * step(${fe}., y) * (.4 + .6 * on * exp(-pow(dz / 3., 2.)));
          } else {
            // the terrace: the lower city's street grid (as buildGround), brighter along the promenade behind the quay
            vec2 q = vec2(vP.x, s);
            vec2 st = abs(fract(q / vec2(45., 60.)) - .5) * vec2(45., 60.);
            float street = exp(-pow(min(st.x, st.y) / 1.6, 2.));
            c = mix(vec3(.17, .14, .15), vec3(.03, .022, .035), uNight) + vec3(1., .55, .3) * street * .35 * mix(.4, 1., uNight);
            c += vec3(1., .7, .45) * .25 * exp(-(vP.x - ${ee+1}.) / 6.);
          }
          float d = length(vP);
          c = mix(c, uFogColor, fogOf(d));
          c = mix(c, vec3(.66, .53, .5), (1. - uNight) * (1. - exp(-d / 240.)) * .5); // the towers' dusk air
          gl_FragColor = lin(c);
        }`}));return d.name="ride-shore",d}const j=(a,e,s,o,i)=>a.push(...e,...s,...o,...e,...o,...i);function pt(a,e,s={}){return $({uniforms:a,side:le,...s,vertexShader:`
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
      }`,fragmentShader:e})}function ga(a,[e,s]){const o=[],i=[],n=[];for(let g=e;g<s;g+=4){const f=o.length;j(o,[-10.3,0,0],[-10.3,6.4,0],[-10.3,6.4,-4],[-10.3,0,-4]),j(o,[5,0,0],[5,0,-4],[5,6.4,-4],[5,6.4,0]),j(o,[-10.3,6.4,0],[5,6.4,0],[5,6.4,-4],[-10.3,6.4,-4]);for(let c=f;c<o.length;c+=3)i.push(g),n.push(c-f<36?0:1);if(Math.round((g-e)/4)%2===0)for(const c of[-10.3+.25,5-.25]){const h=o.length,p=6.4-.9,m=6.4-.6,u=-.4,y=-2.8,z=c+(c<0?.18:-.18);j(o,[z,p,u],[z,m,u],[z,m,y],[z,p,y]),j(o,[c,p,u],[z,p,u],[z,p,y],[c,p,y]);for(let k=h;k<o.length;k+=3)i.push(g),n.push(2)}}for(const[g,f]of[[e,1],[s,-1]]){const c=o.length,h=0;j(o,[-34,0,h],[-10.3,0,h],[-10.3,24,h],[-34,24,h]),j(o,[5,0,h],[30,0,h],[30,24,h],[5,24,h]),j(o,[-10.3,6.4,h],[5,6.4,h],[5,24,h],[-10.3,24,h]);for(let p=c;p<o.length;p+=3)i.push(g),n.push(3);if(f>0){const p=o.length;j(o,[-6,6.4+1.2,h+.05],[2.5,6.4+1.2,h+.05],[2.5,6.4+3,h+.05],[-6,6.4+3,h+.05]);for(let m=p;m<o.length;m+=3)i.push(g),n.push(4)}}const v=new de;v.setAttribute("position",new T(o,3)),v.setAttribute("aS",new T(i,1)),v.setAttribute("aK",new T(n,1));const w=new P(v,pt(a,`
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
    }`));return w.name="ride-tunnel",w}function xa(a,[e,s]){const o=[],i=(f,c,h,p,m,u,y,z)=>{const k=new ye(c-f,p-h,u-m);k.translate((f+c)/2,(h+p)/2,(m+u)/2),k.deleteAttribute("uv"),k.deleteAttribute("normal");const M=k.attributes.position.count;k.setAttribute("aS",new T(new Array(M).fill(y),1)),k.setAttribute("aK",new T(new Array(M).fill(z),1)),o.push(k)},n=e+170,r=s-190,t=92,l=-24,d=8;for(const f of[n,r]){for(const c of[l,d])i(c-1.6,c+1.6,-46,t,-1.6,1.6,f,0),i(c-.5,c+.5,t,t+1.2,-.5,.5,f,2);for(const c of[28,82])i(l,d,c,c+3,-1.2,1.2,f,0)}const v=f=>{if(f<n)return 2+(t-2)*Math.max(0,(f-e)/(n-e))**1.6;if(f>r)return 2+(t-2)*Math.max(0,(s-f)/(s-r))**1.6;const c=(f-n)/(r-n);return t-(t-10)*4*c*(1-c)};for(const f of[l+1,d-1]){for(let c=e;c<s;c+=6){const h=v(c),p=v(c+6),m=new de,u=[];j(u,[f-.35,h,0],[f+.35,h,0],[f+.35,p,-6],[f-.35,p,-6]),j(u,[f,h-.35,0],[f,h+.35,0],[f,p+.35,-6],[f,p-.35,-6]),m.setAttribute("position",new T(u,3)),m.setAttribute("aS",new T(new Array(12).fill(c),1)),m.setAttribute("aK",new T(new Array(12).fill(1),1)),o.push(m)}for(let c=e+12;c<s-6;c+=12){const h=v(c);h<3||i(f-.06,f+.06,1,h,-.06,.06,c,1)}}const w=Ye(o.map(f=>f.index?f.toNonIndexed():f)),g=new P(w,pt(a,`
    varying vec3 vP;
    varying float vS, vK;
    void main() {
      vec3 c;
      if (vK < .5) {
        // towers: pale concrete, darker toward the top, floodlit from the deck, a row of lamps along each crossbeam.
        // The light has a source now: a row of small lamp heads at the foot of each face, aimed up the faces along
        // the road; the faces across the road take only their spill (normal from the flat triangles). The leg the car
        // passes 8 m off used to be one evenly glowing slab right at eye level that bloomed into a white block
        // filling the windscreen's top right (t075); close up its face is held under the bloom threshold.
        vec3 n = normalize(cross(dFdx(vP), dFdy(vP)));
        float face = mix(.55, 1., abs(n.z));
        float wash = smoothstep(.3, 4., vP.y) * exp(-max(vP.y - 4., 0.) / 22.);
        c = vec3(.7, .64, .72) * (.4 + .6 * exp(-max(vP.y, 0.) / 70.)) * face;
        c += vec3(1., .78, .86) * .26 * wash * face;
        c *= mix(.68, 1., smoothstep(20., 80., length(vP)));
        c += vec3(1., .9, .78) * 1.6 * step(.72, fract((vP.x + vP.z) * .9)) * step(-.2, vP.y) * step(vP.y, .45);
        c += vec3(1., .9, .75) * step(.82, fract(vP.x * .5)) * (step(abs(vP.y - 29.5), .25) + step(abs(vP.y - 83.5), .25)) * 2.;
      } else if (vK < 1.5) {
        // cables and hangers: lit by small lamps along them
        c = vec3(.75, .8, 1.) * (.35 + .65 * step(.8, fract(vP.y * .2 + vP.z * .05)));
      } else {
        c = vec3(1., .1, .08) * 4. * step(.5, fract(uT * .6));
      }
      c = mix(c, uFogColor, fogOf(length(vP)) * (vK > 1.5 ? .3 : 1.));
      gl_FragColor = lin(c);
    }`));return g.name="ride-bridge",g}const ya=[["BAY TUNNEL","HARBOUR  2 km"],["BAY BRIDGE","NOCTILUCA BAY"],["EXIT 1 km","MAIN STREET"],["MAIN STREET","KERB  ·  CENTRE"]];function ba(){const a=document.createElement("canvas");a.width=1024,a.height=512;const e=a.getContext("2d");ya.forEach(([o,i],n)=>{const r=n%2*512,t=Math.floor(n/2)*256;e.fillStyle="#0c5a4a",e.fillRect(r,t,512,256),e.strokeStyle="#e8f4ee",e.lineWidth=8,e.strokeRect(r+14,t+14,484,228),e.fillStyle="#f4faf6",e.textAlign="center",e.font='bold 64px "Oxanium", "Big Shoulders Display", sans-serif',e.fillText(o,r+256,t+110),e.font='44px "Oxanium", "Big Shoulders Display", sans-serif',e.fillText(i,r+256,t+186)});const s=new Be(a);return s.colorSpace=Ae,s.anisotropy=4,s}function Sa(a,e){const s=[],o=[],i=[],n=[],r=(w,g,f,c)=>{s.push(...w),o.push(...g),i.push(f),n.push(c)},t=(w,g,f,c,h,p)=>{for(const m of[h,h-.5]){const u=[[w,f,m],[g,f,m],[g,c,m],[w,f,m],[g,c,m],[w,c,m]];for(const y of u)r(y,[0,0],p,0)}};e.forEach(({s:w,board:g})=>{t(-10,-9.6,0,8.2,0,w),t(4.6,5,0,8.2,0,w),t(-10,5,7.6,8.2,0,w);const f=g%2/2,c=1-(Math.floor(g/2)+1)/2,h=-6.8,p=.6,m=4.8,u=7.5,y=.06,z=[[h,m,y,f,c],[p,m,y,f+.5,c],[p,u,y,f+.5,c+.5],[h,m,y,f,c],[p,u,y,f+.5,c+.5],[h,u,y,f,c+.5]];for(const[k,M,I,R,_]of z)r([k,M,I],[R,_],w,1)});const l=new de;l.setAttribute("position",new T(s,3)),l.setAttribute("uv",new T(o,2)),l.setAttribute("aS",new T(i,1)),l.setAttribute("aK",new T(n,1));const d=ba(),v=new P(l,$({uniforms:{...a,uBoards:{value:d}},side:le,vertexShader:`
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
      }`}));return v.name="ride-gantries",v.userData.textures=[d],v}const ut=520,Ma=40;function za(){const a=[],e=(s,o,i,n,r,t)=>{const l=new ye(s,o,i);l.translate(0,n,r),l.deleteAttribute("uv"),l.setAttribute("kind",new T(new Array(l.attributes.position.count).fill(t),1)),a.push(l)};return e(1.82,.62,4.4,.62,0,0),e(1.5,.5,2.2,1.18,.25,1),Ye(a)}function ka(){const a=[],e=[],s=[],o=(r,t,l,d,v,w)=>{const g=[[r,l,v],[t,l,v],[t,d,v],[r,l,v],[t,d,v],[r,d,v]],f=[[0,0],[1,0],[1,1],[0,0],[1,1],[0,1]];g.forEach((c,h)=>{a.push(...c),e.push(...f[h]),s.push(w)})},i=(r,t,l,d,v,w)=>{const g=[[r,l,d],[t,l,d],[t,l,v],[r,l,d],[t,l,v],[r,l,v]],f=[[0,0],[1,0],[1,1],[0,0],[1,1],[0,1]];g.forEach((c,h)=>{a.push(...c),e.push(...f[h]),s.push(w)})};o(-1.1,1.1,.35,.95,-2.22,0),o(-1.1,1.1,.5,1,2.22,1);for(const r of[-.7,.7]){const t=[[r,.55,-2.2],[r,.85,-2.2],[r,.85,18],[r,.55,-2.2],[r,.85,18],[r,.55,18]],l=[[0,0],[0,1],[1,1],[0,0],[1,1],[1,0]];t.forEach((d,v)=>{a.push(...d),e.push(...l[v]),s.push(2)})}i(-1.6,1.6,.03,-2.2,-14,3);const n=new de;return n.setAttribute("position",new T(a,3)),n.setAttribute("uv",new T(e,2)),n.setAttribute("kind",new T(s,1)),n}function Pa(a){const e=[];let s=7;const o=()=>(s=s*16807%2147483647)/2147483647,i=Math.round(a*.4);for(let n=0;n<a;n++){const r=n>=i,t=r?ke.oncoming:[ke.mid,ke.far];e.push({x:t[Math.floor(o()*t.length)]+(o()-.5)*.4,z0:-o()*ut,v:r?18+o()*12:16+o()*10,dir:r?1:-1,paint:o()})}return e}const rt=`
  attribute vec4 aCar; // x, z0, speed, dir (+1 oncoming, -1 our way)
  attribute float aPaint;
  attribute float kind;
  varying vec3 vP, vN, vL;
  varying float vKind, vPaint, vRel, vDir, vS;
  varying vec2 vUv;
  void main() {
    float dir = aCar.w;
    // oncoming cars come toward us at their own speed plus ours; our way, they move ahead of us at theirs
    float z = wrapZ(aCar.y + dir * aCar.z * uT, ${ut}., ${Ma}.);
    vec3 p = position;
    vL = position; // car space (front at -z), for the panels and the glass
    vN = normal;
    if (dir > 0.) { p.x = -p.x; p.z = -p.z; vN.xz = -vN.xz; }
    float s = uD - z;
    p.x += aCar.x; p.z += z;
    vRel = dir > 0. ? aCar.z + uSpeed : abs(uSpeed - aCar.z);
    vKind = kind; vPaint = aPaint; vDir = dir; vS = s; vP = p;
    #ifdef USE_UV
    vUv = uv;
    #endif
    gl_Position = projectionMatrix * modelViewMatrix * vec4(bend(p), 1.);
  }`;function lt(a,e){const s=new Ne;a.index&&(s.index=a.index);for(const n of Object.keys(a.attributes))s.setAttribute(n,a.attributes[n]);const o=new Float32Array(e.length*4),i=new Float32Array(e.length);return e.forEach((n,r)=>{o.set([n.x,n.z0,n.v,n.dir],r*4),i[r]=n.paint}),s.setAttribute("aCar",new ae(o,4)),s.setAttribute("aPaint",new ae(i,1)),s.instanceCount=e.length,s}function Aa(a,e,{trails:s=!0}={}){const o=new se;o.name="ride-traffic";const i=new P(lt(za(),e),$({uniforms:a,vertexShader:rt,fragmentShader:`
        varying vec3 vP, vN, vL;
        varying float vKind, vPaint, vRel, vDir, vS;
        void main() {
          vec3 paint = vPaint < .3 ? vec3(.09, .09, .11) : vPaint < .55 ? vec3(.5, .52, .58) : vPaint < .8 ? vec3(.24, .06, .1) : vec3(.05, .1, .2);
          float lamp = exp(-pow((mod(vS, 32.) - 2.) / 8., 2.)) * (1. - inTunnel(vS));
          vec3 n = normalize(vN);
          float up = step(.5, n.y), side = step(.5, abs(n.x)), endF = step(.5, abs(n.z));
          vec3 sky = mix(vec3(.16, .11, .19), vec3(.3, .22, .26), 1. - uNight); // what a roof reflects
          vec3 c;
          if (vKind < .5) {
            // body: roof-facing panels lit from above, sides in shade with a shoulder highlight, ends darkest
            c = paint * (.2 + .55 * lamp) + vec3(.05, .035, .06);
            c += up * (paint * (.25 + .8 * lamp) + sky * .6 + vec3(.25, .2, .28) * lamp * .35);
            float shoulder = exp(-pow((vL.y - .86) / .035, 2.));
            c += side * shoulder * (sky * 1.1 + vec3(.3, .24, .3) * lamp * .5);
            c *= mix(1., .55, side * (1. - smoothstep(.34, .55, vL.y))); // the sill falls into shadow
            c *= mix(1., .75, endF);
            // wheels in their arches on the sides
            float wz = abs(abs(vL.z) - 1.35);
            float wr = length(vec2(wz, vL.y - .33));
            float wheel = side * (1. - smoothstep(.31, .34, wr));
            vec3 tyre = mix(vec3(.025, .025, .03), vec3(.16, .16, .19) + .2 * lamp, smoothstep(.13, .11, wr) * step(.06, wr));
            c = mix(c, tyre, wheel);
          } else {
            // glasshouse: the roof is paint, the rest pillars and glass
            float glass = side * step(.98, vL.y) * step(vL.y, 1.36)
              * (step(-.78, vL.z) * step(vL.z, .2) + step(.3, vL.z) * step(vL.z, 1.22))
              + endF * step(abs(vL.x), .64) * step(.98, vL.y) * step(vL.y, 1.38);
            vec3 pillar = paint * (.22 + .5 * lamp) + vec3(.04, .03, .05);
            float gy = clamp((vL.y - .98) / .4, 0., 1.);
            // the glass holds the sky above and a dim band of the city's lights at the horizon line
            vec3 g = mix(vec3(.025, .025, .04), sky * .8, gy * gy) + vec3(.35, .25, .35) * lamp * .18;
            g += vec3(.22, .16, .2) * exp(-pow((gy - .42) / .08, 2.)) * (side + .5 * endF);
            c = mix(pillar, g, clamp(glass, 0., 1.));
            c = mix(c, paint * (.3 + .9 * lamp) + sky * .6 + vec3(.25, .2, .28) * lamp * .35, up);
          }
          c *= mix(vec3(1.), vec3(1.2, .85, .55), inTunnel(vS));
          c = mix(c, uFogColor, fogOf(length(vP)));
          gl_FragColor = lin(c);
        }`}));i.name="ride-traffic-bodies",o.add(i);const n=ka(),r=new P(lt(n,e),$({uniforms:{...a,uTrails:{value:s?1:0}},transparent:!0,depthWrite:!1,blending:ft,side:le,defines:{USE_UV:""},vertexShader:rt,fragmentShader:`
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
        }`}));return r.name="ride-traffic-lamps",r.renderOrder=3,o.add(r),o}const C={x:.36,z0:-.36,z1:.38,y:1.33};function Ta(a){a.updateMatrixWorld(!0);const e=new dt,s=new re,o=new re;a.traverse(i=>{if(!i.isMesh||i.name)return;const n=i.material;n!=null&&n.isMeshBasicMaterial&&n.transparent&&(e.setFromObject(i).getSize(s),e.getCenter(o),s.x<.05&&s.z>2.4&&s.z<2.8&&Math.abs(o.x)>.7&&o.y>1&&o.y<1.4&&(i.name=o.x>0?"glass-right":"glass-left"))})}function Ea(a,e,s){let o=null,i=1/0;const n=new re;return a.traverse(r=>{if(r.name!==e)return;r.getWorldPosition(n);const t=s?n.distanceTo(s):0;t<i&&(i=t,o=r)}),o}function Ra(a,e){const s=[],o=new Map,i=new dt;return a.updateMatrixWorld(!0),a.traverse(n=>{if(!n.isMesh||n.userData.ride||n.userData.noCollide||[].concat(n.material).some(t=>t==null?void 0:t.isShaderMaterial)||(i.setFromObject(n),i.max.y<C.y||i.min.x>C.x||i.max.x<-.36||i.min.z>C.z1||i.max.z<C.z0))return;const r=t=>{if(!t||t.transparent)return t;if(!o.has(t)){const l=t.clone();l.onBeforeCompile=d=>{d.uniforms.uRoofOpen=e,d.vertexShader=d.vertexShader.replace("void main() {",`varying vec3 vRoofW;
void main() {`).replace("#include <project_vertex>",`#include <project_vertex>
  vRoofW = (modelMatrix * vec4(transformed, 1.0)).xyz;`),d.fragmentShader=d.fragmentShader.replace("void main() {",`uniform float uRoofOpen;
varying vec3 vRoofW;
void main() {
  if (uRoofOpen > 0.001 && vRoofW.y > ${C.y.toFixed(3)} && abs(vRoofW.x) < ${C.x.toFixed(3)} && vRoofW.z > ${C.z0.toFixed(3)} && vRoofW.z < ${C.z0.toFixed(3)} + ${(C.z1-C.z0).toFixed(3)} * uRoofOpen) discard;`)},l.customProgramCacheKey=()=>"ride-roof",o.set(t,l),s.push(l)}return o.get(t)};n.material=Array.isArray(n.material)?n.material.map(r):r(n.material)}),s}function Fa(){const a=document.createElement("canvas");a.width=512,a.height=224;const e=new Be(a);e.colorSpace=Ae;const s=new P(new be(.31,.13),new Pe({map:e,toneMapped:!1}));s.position.set(-.012,.983,-.9115),s.rotation.x=-.08,s.name="ride-screen";let o="";return{mesh:s,draw:n=>{const r=JSON.stringify(n);if(r===o)return;o=r;const t=a.getContext("2d");t.fillStyle="#050b14",t.fillRect(0,0,512,224),t.fillStyle="rgba(120,170,230,0.5)",t.fillRect(12,11,488,1),t.font="11px sans-serif",t.fillStyle="#7fa8d8",t.fillText(n.clock,15,25),t.fillText("MEDIA   ROUTE   CAR",190,25);const l=n.station>0;if(t.fillStyle=l?"#13243a":"#0b1420",t.fillRect(14,38,200,172),t.fillStyle=l?"#ff9ab8":"#4a6a8a",t.font="bold 15px sans-serif",t.fillText(Le[n.station].name,24,62),t.font="12px sans-serif",t.fillStyle="#9cc4ee",l){t.fillText(Le[n.station].track,24,84),t.fillText(Le[n.station].artist,24,102);for(let d=0;d<14;d++){const v=8+(d*37+n.beat*13)%40;t.fillStyle="#4aa8ff",t.fillRect(24+d*13,190-v,8,v)}}else t.fillText("F at the screen: tune in",24,84);t.strokeStyle="#2a4a6a",t.lineWidth=4,t.beginPath(),t.moveTo(250,120),t.lineTo(490,120),t.stroke(),t.strokeStyle="#4aa8ff",t.beginPath(),t.moveTo(250,120),t.lineTo(250+240*n.progress,120),t.stroke(),ht.forEach((d,v)=>{const w=250+240*d.t/H;t.fillStyle=n.chapter>=d.n?"#ff9ab8":"#2a4a6a",t.beginPath(),t.arc(w,120,6,0,Math.PI*2),t.fill(),t.fillStyle="#7fa8d8",t.font="10px sans-serif",t.fillText(String(v+1),w-3,142)}),t.fillStyle="#ffffff",t.beginPath(),t.arc(250+240*n.progress,120,8,0,Math.PI*2),t.fill(),n.done?(t.strokeStyle="#ff9ab8",t.lineWidth=3,t.beginPath(),t.moveTo(252,72),t.lineTo(259,80),t.lineTo(273,62),t.stroke(),t.fillStyle="#ff9ab8",t.font="bold 22px sans-serif",t.fillText(n.title,282,80,216)):(t.fillStyle="#9cc4ee",t.font="bold 16px sans-serif",t.fillText(n.title,250,80,248)),t.fillStyle="#9cc4ee",t.font="14px sans-serif",t.fillText(n.eta,250,178,248),e.needsUpdate=!0},tex:e}}function Ca(){const a=document.createElement("canvas");a.width=512,a.height=224;const e=new Be(a);e.colorSpace=Ae;const s=new P(new be(.33,.13),new Pe({map:e,toneMapped:!1}));s.position.set(-.39,.985,-.9115),s.rotation.x=-.08,s.name="ride-cluster";let o="";return{mesh:s,draw:(n,r)=>{const t=`${Math.round(n)}${r}`;if(t===o)return;o=t;const l=a.getContext("2d");l.setTransform(.5,0,0,.5,0,0),Wt(l,1024,448,Math.round(n),r),e.needsUpdate=!0},tex:e}}function Da(){const a=new se;a.name="ride-charm";const e=new P(new Ve(.0012,.0012,.11,4),new Pe({color:10127984}));e.position.y=-.055;const s=new Dt;s.absarc(0,0,.032,Math.PI*.15,Math.PI*1.85,!1),s.absarc(.014,0,.026,Math.PI*1.75,Math.PI*.25,!0);const o=new P(new Lt(s,{depth:.003,bevelEnabled:!1,curveSegments:18}),new We({color:15918792,emissive:9069624,emissiveIntensity:.5,roughness:.9,side:le}));return o.position.set(0,-.14,-.0015),o.rotation.z=.25,a.add(e,o),a.position.set(.09,1.262,-.63),a}const La=[{rough:.38,set:{roughness:.74,color:4605003}},{rough:.48,set:{roughness:.64,color:3420986}},{rough:.5,set:{roughness:.7,color:2763055}}];function Oa(a){const e=new Map;return a.traverse(s=>{const o=s.material;if(!s.isMesh||!(o!=null&&o.isMeshStandardMaterial)||o.metalness>.05)return;const i=La.find(n=>Math.abs((o.userData.authorRough??o.roughness)-n.rough)<.015);if(i){if(!e.has(o)){const n=o.clone();o.userData.finish||(n.roughness=i.set.roughness),n.color.setHex(i.set.color),e.set(o,n)}s.material=e.get(o)}}),[...e.values()]}function Na({practicals:a=!0}={}){const e=new se;e.name="ride-lights";const s=new Ct(4868704,2102296,.34),o=new Ge(16758984,1.3);o.position.set(20,31,-30),o.target.position.set(0,1,0),o.castShadow=!0,o.shadow.mapSize.set(2048,2048),Object.assign(o.shadow.camera,{left:-4,right:4,top:4,bottom:-4,near:38,far:56}),o.shadow.bias=-4e-4,o.shadow.normalBias=.006;const i=new Fe(16761032,60,16,1.4);i.position.set(1.5,7,-20);const n=new Fe(16730832,30,9,1.6);n.position.set(-7,2,-12);const r=new Ge(10131104,.52);r.position.set(-.2,1.7,1),r.target.position.set(.1,.85,-1);const t=new Ge(16042144,0);t.position.set(-1.6,2.1,-.9),t.target.position.set(.5,.95,.2);const l=new Fe(8688832,.1,.8,2);l.position.set(-.2,.95,-.6),e.add(s,o,o.target,i,n,r,r.target,t,t.target,l);const d=[];if(a){for(const[v,w,g,f,c]of[[.655,.688,-.42,.018,.34],[.655,.688,-.08,.018,.34],[.655,.688,.26,.015,.34],[.6,.672,-.815,.014,.3]]){const h=new Fe(16754786,f,c,2);h.position.set(v,w,g),d.push(h)}e.add(...d)}return{group:e,hemi:s,key:o,pool:i,mag:n,fill:r,win:t,disp:l,amber:d}}function Ba(a,e){const s=new se;s.name="ride-cabin";const o={value:0},i=Ra(a,o),n=a.getObjectByName("sunroof");n&&(n.visible=!1);const r=new We({color:11845078,roughness:.32,metalness:.35,emissive:1316900}),t=new se;t.name="ride-sunroof";const l=C.x*2,d=C.z1-C.z0,v=(C.z0+C.z1)/2;for(const[S,B,he,ce]of[[l+.06,.03,0,C.z0-.015],[l+.06,.03,0,C.z1+.015],[.03,d,-.36-.015,v],[.03,d,C.x+.015,v]]){const Se=new P(new ye(S,.012,B),r);Se.position.set(he,1.392,ce),t.add(Se)}const w=new Pe({color:3816026,transparent:!0,opacity:.35,depthWrite:!1,side:le}),g=new P(new be(l,d),w);g.rotation.x=Math.PI/2,g.position.set(0,1.455,v),s.add(t,g);const f=Da();s.add(f);const c=Fa();s.add(c.mesh);const h=Ca();s.add(h.mesh);const p=new Pe({color:3866522,toneMapped:!1}),m=new We({color:2764344,roughness:.45,metalness:.5}),u=new se;u.name="ride-lightswitch",u.position.set(-.72,.79,-.858);const y=new P(new Ve(.024,.026,.022,20).rotateX(Math.PI/2),m);y.position.z=.011;const z=new P(new ye(.004,.018,.004),p);z.position.set(0,.01,.023),y.add(z);const k=new P(new Rt(.005,12),p);k.position.set(.04,.012,.002),u.add(y,k),s.add(u),i.push(r,w,p,m,c.mesh.material,c.tex,h.mesh.material,h.tex);for(const S of[f,t,g,c.mesh,h.mesh,u])S.traverse(B=>{B.userData.ride=!0,B.raycast=()=>{}});s.traverse(S=>{S.userData.noCollide=!0});const M=a.getObjectByName("wheel"),I=M==null?void 0:M.quaternion.clone();M&&i.push(...Oa(M));const R=Ea(a,"glass-right",new re(1,1.2,0)),_=(R==null?void 0:R.position.y)??0,U=a.getObjectByName("leadcar"),V=U==null?void 0:U.position.clone(),te=new re(0,0,1),ne=new Ft,Y={t0:-99,amp:0};return{group:s,owned:i,uniforms:{uRoofOpen:o},flick(S){Y.t0=S,Y.amp=.75},screen:c,update(S,B){o.value=S.roof,h.draw(S.speed*3.6,S.t>H-.5?"P":"D"),g.position.z=v+d*S.roof*.98,g.visible=S.roof>.01,R&&(R.position.y=_-.44*S.window),M&&M.quaternion.copy(I).multiply(ne.setFromAxisAngle(te,E.clamp(-S.curve*900,-.5,.5)));const he=B-Y.t0,ce=Y.amp*Math.exp(-he/2.2)*Math.sin(he*5.2);f.rotation.x=ce+S.accel*.03+Math.sin(B*1.7)*.015*Math.min(1,S.speed/10),f.rotation.z=E.clamp(S.curve*S.speed*S.speed*.5,-.4,.4)+ce*.2,p.color.setHex(S.head?3866522:1058330),y.rotation.z=S.head?-.9:0,U&&(U.position.x=V.x-1+Math.sin(B*.31)*.12,U.position.z=V.z-Math.max(0,S.t-(H-14))**2*.9,U.visible=U.position.z>-400)},dispose(){var S;for(const B of i)(S=B.dispose)==null||S.call(B)}}}const ct={high:{traffic:46,rain:1400,signs:!0,midRow:!0,farRow:!0,facades:!0,trails:!0,practicals:!0},medium:{traffic:34,rain:800,signs:!0,midRow:!0,farRow:!0,facades:!0,trails:!0,practicals:!0},low:{traffic:24,rain:360,signs:!0,midRow:!1,farRow:!1,facades:!1,trails:!1},software:{traffic:14,rain:0,signs:!1,midRow:!1,farRow:!1,facades:!1,trails:!1}},Ce={from:"kouka",title:"Kouka",text:"Saw your car come out of the bay tunnel on the traffic feed. The water is glowing again tonight, the plankton bloom. Look left before the bridge ends."},De=a=>{try{return new URLSearchParams(location.search).get(a)}catch{return null}};function Ka(a){var s,o;const e=a.getObjectByName("leadcar");e&&e.parent!==a&&a.attach(e);for(const i of["road","wet","streaks","lights","skyDome"]){const n=a.getObjectByName(i);n&&(n.visible=!1)}a.traverse(i=>{var n;(n=i.userData)!=null&&n.update&&/mist/i.test(i.name||"")&&i.parent===a&&(i.visible=!1)});for(const i of a.children)i.isMesh&&((o=(s=i.material)==null?void 0:s.uniforms)!=null&&o.uSpeed)&&i!==e&&(i.visible=!1)}function Ga(a,{quality:e}={}){const s=me.tier||(e==="low"?"low":"high"),o=ct[s]||ct.high,i=De("tod")?De("tod")!=="day":Ot(),n=De("rain")!=="0",r=a.root;Ta(r),Ka(r),a.walk={seated:!0,bounds:[-.3,-.3,.3,.3],eye:1.2,floorY:()=>0,colliders:[],start:{position:[0,1.2,.02],target:[0,1.06,-1]},fov:64},a.env=a.env||{},a.env.fog=i?{color:1709104,density:.0026}:{color:6178386,density:.0024},a.env.background=i?395278:3811908;const t=Gt();t.uTunnel.value.set(...N.tunnel),t.uBridge.value.set(...N.bridge),t.uRamp.value=N.ramp,t.uEnd.value=N.end,t.uNight.value=i?1:0,t.uFogColor.value.setHex(a.env.fog.color,Nt),t.uFogDensity.value=a.env.fog.density;const l=new se;l.name="ride";const d=fa({midRow:o.midRow,farRow:o.farRow,signs:o.signs}),v=[],w=x=>{x&&(l.add(x),v.push(...x.userData.textures||[]))};w(Vt(t)),w(ma(t)),w(wa(t)),w(Zt(t)),w(Qt(t)),w(ha(t,d,{facades:o.facades})),w(ua(t,d)),w(ga(t,N.tunnel)),w(xa(t,N.bridge)),w(Sa(t,[{s:420,board:0},{s:N.tunnel[0]-170,board:0},{s:N.bridge[0]+90,board:1},{s:N.bridge[1]-260,board:2},{s:N.ramp+60,board:3}])),w(Aa(t,Pa(o.traffic),{trails:o.trails})),w(qt(t,n?o.rain:0)),st(l),r.add(l);const g={uniforms:{uWipe:{value:0},uWipeRate:{value:0}}},f=Ba(r),c=st(jt(t,g)),h=Ht();h.group.traverse(x=>{x.raycast=()=>{},x.userData.noCollide=!0});const p=Na({practicals:!!o.practicals});r.add(f.group,c,h.group,p.group);const m=Kt(),u={t:0,chapter:1,station:0,roof:0,window:0,wipers:0,head:!0,arrived:!1,rain:0},y={roof:0,window:0},z=new Set,k=new Set,M=x=>{for(const D of k)try{D(x)}catch(K){console.error(K)}};let I=performance.now(),R=0,_=!1;const U=()=>{I=performance.now()};document.addEventListener("visibilitychange",U);const V=(x,D={})=>M({kind:"toast",text:x,...D}),te={"radio:next":()=>{m.arm();const x=m.setStation(m.stationIndex+1===Le.length?0:m.stationIndex+1);return u.station=m.stationIndex,M({kind:"radio",station:u.station}),x.url?`${x.name} · "${x.track}", ${x.artist}`:"The radio clicks off. Just the road now."},"radio:off":()=>(m.arm(),m.setStation(0),u.station=0,M({kind:"radio",station:0}),"The radio clicks off. Just the road now."),"sunroof:toggle":()=>(m.arm(),m.click(),y.roof=y.roof>.5?0:1,y.roof?"The sunroof slides back. Cold air, and the towers leaning in overhead.":"The sunroof slides shut. The city sound drops away."),"window:toggle":()=>(m.arm(),m.click(),y.window=y.window>.5?0:1,y.window?"The window winds down. The wind comes in loud, smelling of rain and the sea.":"The window winds up. Quiet again."),"wipers:cycle":()=>(m.arm(),m.click("wiper"),u.wipers=(u.wipers+1)%3,u.wipers===0&&(_=!0),["Wipers off.","Wipers on.","Wipers fast."][u.wipers]+(u.rain<.05&&u.wipers?" The glass is dry; they squeak once and keep time.":"")),"lights:toggle":()=>(m.arm(),m.click(u.head?"off":"switch"),u.head=!u.head,u.head?"Headlamps on. The lane comes back out of the dark.":"Headlamps off. Between the street lamps the road goes dark; the car does not mind."),"charm:flick":()=>(f.flick(ne()),"You flick the paper moon. It spins on its thread, and the dashboard light runs around its edge."),"route:show":()=>{const x=ot(u.t),D=Math.max(0,Math.round(H-u.t));return u.arrived?"Main street, at the kerb. The route screen says: you have arrived.":`Chapter ${x.n} of 3, ${x.title}. Main street in ${Math.floor(D/60)}:${String(D%60).padStart(2,"0")}.`},"ride:restart":()=>(Y.seek(0),z.clear(),"The car pulls away from the kerb again."),horn:()=>(m.arm(),m.click(),null)},ne=()=>performance.now()/1e3,Y={get state(){return{...u,length:je,end:H}},message:Ce,subscribe(x){return k.add(x),()=>k.delete(x)},use(x){var L,O,Q,A;const D=(Q=(O=(L=x==null?void 0:x.interactable)==null?void 0:L.primary)==null?void 0:O.effect)==null?void 0:Q.device;if(!D||!te[D])return!1;const Z=te[D]()||((A=x.lines)==null?void 0:A[0])||null;return Z&&V(Z,{device:D}),!0},seek(x){u.t=Math.max(0,Math.min(je,x)),I=performance.now(),x<qe.mid&&z.delete("mid"),x<H&&(z.delete("arrive"),u.arrived=!1)},audio:m};a.ride=Y;const S={pos:new re,rx:0,ry:0,rz:0,on:!1,p0:new re,r0:new Bt},B=(x,D,K,Z)=>{const L=performance.now(),O=Math.min(1,(L-I)/1e3);I=L;const Q=document.hidden||!!me.hold||!!me.settling;Q||(u.t=Math.min(je,u.t+O));const A=u.t,q=It(A),J=He(A),ve=Ut(A),pe=(He(A+.25)-He(A-.25))/.5;t.uT.value=L/1e3%3600,t.uD.value=q,t.uSpeed.value=J,t.uCurve.value=ve,t.uHill.value=$t(A),t.uHeading.value=_t(A),u.rain=n?E.clamp((A-qe.rain)/10,0,1)*(A>H+2?.8:1):0,t.uRain.value=u.rain,t.uHead.value=E.lerp(t.uHead.value,u.head?1:0,Math.min(1,O*6)),u.roof+=E.clamp(y.roof-u.roof,-O*.7,O*.7),u.window+=E.clamp(y.window-u.window,-O*.6,O*.6);const Ie=[0,.75,1.35][u.wipers];if(Ie)R+=O*Ie,_=!1;else if(_){const ie=Math.ceil(R);R=Math.min(ie,R+O*.75),R>=ie&&(_=!1)}g.uniforms.uWipe.value=R,g.uniforms.uWipeRate.value=Ie||(_?.75:0);const Ue=R%1,gt=Ue<.5?Ue*2:2-Ue*2;for(const ie of h.blades)ie.rotation.z=gt*Xt;f.update({t:A,roof:u.roof,window:u.window,curve:ve,speed:J,accel:pe,head:u.head},L/1e3);const ue=ot(A),$e=Math.max(0,Math.round(H-A)),Ee=u.arrived||$e===0;f.screen.draw({station:u.station,beat:u.station?Math.floor(L/500):0,chapter:ue.n,done:Ee,title:Ee?"Arrived":`${ue.n} · ${ue.title}`,progress:Ee?1:Math.min(1,A/H),eta:Ee?"Main street, at the kerb":`Main street ${Math.floor($e/60)}:${String($e%60).padStart(2,"0")}`,clock:new Date().toTimeString().slice(0,5)});const G=q>N.tunnel[0]&&q<N.tunnel[1]?1:0,Me=q>N.bridge[0]+40&&q<N.bridge[1]-40?1:0,xt=q%oe/oe,Re=Math.exp(-(((xt-.5)*oe/7)**2)),Qe=Math.exp(-(((q%8-4)/2.2)**2)),{key:Je,hemi:_e,pool:yt,mag:et,fill:tt,win:bt}=p;Je.intensity=G?.6+.9*Qe:(Me?.95:1.1)+.35*Re,Je.color.setHex(G?16754768:Me?12110079:i?16758984:16765620),_e.intensity=G?.2+.35*Qe:i?Me?.42:.3+.22*Re:.5+.1*Re,_e.color.setHex(G?16752736:i?4999256:8813700),bt.intensity=i||G?0:.95,_e.groundColor.setHex(Me?932410:G?3810320:2102296),yt.intensity=(G?25:60)*(u.head?1:.35);const St=Math.floor(q/18)%3;et.color.setHex(G?16748608:Me?4247776:[16730832,4905215,16742970][St]),et.intensity=G?10:30;const Mt=(i?.52:.85)*(G?.85:me.still?1:.94+.16*Re);if(tt.intensity=E.lerp(tt.intensity,Mt,Math.min(1,O*8)),m.update({speed:J,wind:Math.max(u.window,u.roof*.6),rain:u.rain,tunnel:G,paused:Q}),Z!=null&&Z.warp&&mt(Z.warp,{v:J,accel:pe,tun:G,D:q,camera:K}),(ue.n!==u.chapter||!z.has("chapter1"))&&(u.chapter=ue.n,z.add("chapter1"),M({kind:"chapter",chapter:ue})),!z.has("start")&&A>2.5&&(z.add("start"),Xe("ride:start",{place:"car-1",chapters:ht.length,length:H})),!z.has("mid")&&A>=qe.mid&&(z.add("mid"),Xe("ride:mid",{place:"car-1",where:"bay-bridge",hie:Ce.from,message:Ce.text}),M({kind:"message",message:Ce})),!z.has("arrive")&&A>=H&&(z.add("arrive"),u.arrived=!0,Xe("ride:arrive",{place:"car-1",to:"street-1"}),M({kind:"arrive"})),K&&!me.still){const ie=Math.min(1,J/20);S.pos.set(Math.sin(L*.0021)*.002*ie,(Math.sin(L*.0083)*.0022+Math.sin(L*.0191)*.0011)*ie+(wt(L)-.5)*6e-4*ie,0),S.rz=E.clamp(-ve*J*J*.035,-.035,.035),S.ry=E.clamp(ve*18,-.03,.03),S.rx=E.clamp(-pe*.004,-.02,.02),S.p0.copy(K.position),S.r0.copy(K.rotation),K.position.add(S.pos),K.rotation.x+=S.rx,K.rotation.y+=S.ry,K.rotation.z+=S.rz,S.on=!0}},he=Number(De("warpBridge")??.2),[ce,Se]=N.bridge,Te=new re,mt=(x,{v:D,accel:K,tun:Z,D:L,camera:O})=>{const Q=me.still||s==="software"||!O,A=E.smoothstep(D,17,25),q=E.clamp(K/3,0,1),J=L-N.tunnel[1],ve=E.smoothstep(L,ce+40,ce+140)*(1-E.smoothstep(L,Se-120,Se-40));if(x.amount=Q?0:Math.min(.85,E.lerp(.5,he,ve)*A+.3*q+.25*Z),x.lines=Q?0:Math.min(.6,E.smoothstep(K,.3,1)*A+.35*Z),x.flash=Q||J<0?0:.8*Math.exp(-J/10),x.near=2.5,!Q){const pe=E.smoothstep(-O.getWorldDirection(Te).z,.35,.75);x.amount*=pe,x.lines*=pe,Te.set(0,1.2,-100).project(O),x.center[0]=E.clamp(Te.x*.5+.5,0,1),x.center[1]=E.clamp(Te.y*.5+.5,0,1)}},wt=x=>{const D=Math.sin(x*12.9898)*43758.5453;return D-Math.floor(D)};a.update=B,a.afterRender=x=>{S.on&&(x.position.copy(S.p0),x.rotation.copy(S.r0),S.on=!1)};const Ke=a.dispose;return a.dispose=()=>{Ke==null||Ke(),document.removeEventListener("visibilitychange",U),m.dispose(),f.dispose(),k.clear(),l.traverse(x=>{x.isMesh&&(x.geometry.dispose(),x.material.dispose())});for(const x of v)x.dispose();c.geometry.dispose(),c.material.dispose(),h.material.dispose()},Y}export{Ga as attachRide};
