import{G as ne,e as A,O as xt,U as Xe,a0 as Je,Z as we,$ as T,a5 as Ee,a6 as ae,P as xe,X as ce,L as me,b6 as gt,S as Me,aa as yt,aj as bt,af as St,aF as Mt,C as Fe,y as it,d as rt,F as re,a4 as je,M as Se,ac as zt,bF as X,ao as kt,bG as O,R as Pt,T as Ke,Q as Ie,a2 as At,ak as Tt,bH as Te,bI as lt,c as he,bJ as Rt,bK as L,bL as Et,as as Ft,bM as Ue,bN as _e,bO as Ct,bP as Dt,bQ as Ge,bR as Ot,bS as Lt,bT as Nt,bU as et,bV as $e}from"./index-DClJX3kH.js";import{ax as $,ay as Bt,az as Kt,aA as tt,aB as It,aC as Ut,aD as _t}from"./texlocal-Cz2HTPrJ.js";import{paintCluster as Gt}from"./ctex-D4egpL11.js";import"./react-three-fiber.esm-pOii4C3L.js";function $t(a){const e=new ne;e.name="ride-sky";const o=new A(new xt(900,32,16),$({uniforms:a,side:Je,depthWrite:!1,vertexShader:`
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
        }`}));o.renderOrder=-10,e.add(o);const n=new A(new Xe(650,650,260,96,1,!0),$({uniforms:a,side:Je,transparent:!0,depthWrite:!1,vertexShader:`
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
        }`}));return n.renderOrder=-9,e.add(n),e}const be={mid:-3.6,far:-7.2,oncoming:[-12.2,-15.8,-19.2],parapet:4.6,median:-9.8},oe=32,pe=760,ye=14;function Ht(a,e,o=190){const n=new xe(e-a,pe,1,o);return n.rotateX(-Math.PI/2),n.translate((a+e)/2,0,ye-pe/2),n}function jt(a){const e=new ne;e.name="ride-road";const o=new A(Ht(-22,5.2),$({uniforms:a,vertexShader:`
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
        }`}));o.name="ride-road-deck",e.add(o);const n=new we,i=[],s=[],r=(d,v,w,x)=>{for(let c=0;c<190;c++){const h=ye-pe*c/190,p=ye-pe*(c+1)/190;i.push(d,v,h,d,w,h,d,v,p,d,w,h,d,w,p,d,v,p);for(let m=0;m<6;m++)s.push(x)}},t=(d,v,w,x)=>{for(let c=0;c<190;c++){const h=ye-pe*c/190,p=ye-pe*(c+1)/190;i.push(d,w,h,v,w,h,d,w,p,v,w,h,v,w,p,d,w,p);for(let m=0;m<6;m++)s.push(x)}};r(-9.4,0,.85,0),r(-10.2,0,.85,0),t(-10.2,-9.4,.85,1),r(4.4,0,1.15,2),t(4.4,4.9,1.15,3),r(4.9,-1.2,1.15,2),r(-22,-1.2,.9,2),r(5.2,-1.4,0,4),n.setAttribute("position",new T(i,3)),n.setAttribute("kind",new T(s,1));const l=new A(n,$({uniforms:a,side:ce,vertexShader:`
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
        }`}));return l.name="ride-barriers",e.add(l),e}function Xt(a){const e=[],o=(d,v,w,x,f,c,h)=>{const p=new me(d,v,w);p.translate(x,f,c),p.setAttribute("glow",new T(new Array(p.attributes.position.count).fill(h),1)),p.deleteAttribute("uv"),e.push(p)};o(.22,10,.22,0,5,0,0),o(2.8,.14,.14,-1.4,9.8,0,0),o(.9,.12,.38,-2.6,9.68,0,1);const n=We(e),i=Math.ceil(760/oe),s=i*2,r=new Ee;r.index=n.index;for(const d of Object.keys(n.attributes))r.setAttribute(d,n.attributes[d]);const t=new Float32Array(s*3);for(let d=0;d<i;d++)t.set([be.parapet+.2,-d*oe+2,1],d*3),t.set([be.median,-d*oe-14,-1],(i+d)*3);r.setAttribute("slot",new ae(t,3)),r.instanceCount=s;const l=new A(r,$({uniforms:a,vertexShader:`
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
        }`}));return l.name="ride-lamps",l}function We(a){const e=Object.keys(a[0].attributes),o=new we;let n=0;const i=[],s=Object.fromEntries(e.map(r=>[r,[]]));for(const r of a){const t=r;for(const l of e)s[l].push(...t.attributes[l].array);if(t.index)for(const l of t.index.array)i.push(l+n);else for(let l=0;l<t.attributes.position.count;l++)i.push(l+n);n+=t.attributes.position.count}for(const r of e)o.setAttribute(r,new T(s[r],a[0].attributes[r].itemSize));return o.setIndex(i),o}const ve={GLASS:1,OFFICE:3};function Wt(a){let e=a>>>0;return function(){e=e+1831565813>>>0;let n=e;return n=Math.imul(n^n>>>15,n|1),n^=n+Math.imul(n^n>>>7,n|61),((n^n>>>14)>>>0)/4294967296}}const b=(a,e,o)=>e+(o-e)*a();function qt(a,e){for(let o=e.length-1;o>0;o--){const n=Math.floor(a()*(o+1));[e[o],e[n]]=[e[n],e[o]]}return e}const Vt=(a,e=0)=>Wt(Math.floor(a*2**31)^e*2654435761),W=(a,e,o,n,i,s,r={})=>({x:a,z:e,w:o,d:n,y0:i,y1:s,...r}),E=(a,e,o,n,i,s,r,t={})=>({x:a,z:e,w:o,d:n,y0:i,y1:s,kind:r,feat:!0,...t});function Re(a,e=.6,o=.7){return E(a.x,a.z,a.w+e*2,a.d+e*2,a.y1-o,a.y1,"clad")}function ue(a,e,o,n){const i=a(),s=e.y1;if(i<.3){const r=b(a,9,n?26:16),t=Math.min(1.6,e.w*.08);for(const[l,d]of[[-1,-1],[1,-1],[-1,1],[1,1]])o.push(E(e.x+l*(e.w/2-t/2),e.z+d*(e.d/2-t/2),t,t,s,s+r,"clad"));o.push(E(e.x,e.z-e.d/2+t/2,e.w,t,s+r-1.4,s+r,"led")),o.push(E(e.x,e.z+e.d/2-t/2,e.w,t,s+r-1.4,s+r,"led")),o.push(E(e.x-e.w/2+t/2,e.z,t,e.d-2*t,s+r-1.4,s+r,"led")),o.push(E(e.x+e.w/2-t/2,e.z,t,e.d-2*t,s+r-1.4,s+r,"led")),o.push(E(e.x,e.z,e.w*.45,e.d*.45,s,s+b(a,4,7),"plant"))}else if(i<.55){const r=b(a,6,9);o.push(E(e.x,e.z,e.w-1.2,e.d-1.2,s,s+r,"louvre"));const t=b(a,.62,.8);o.push(E(e.x,e.z,e.w*t,e.d*t,s+r,s+r+b(a,8,n?22:12),"led"))}else if(i<.8){let r=e.w,t=e.d,l=s;const d=2+Math.floor(a()*2);for(let v=0;v<d;v++){r*=b(a,.68,.82),t*=b(a,.68,.82);const w=b(a,5,10);o.push(E(e.x,e.z,r,t,l,l+w,v===d-1?"led":"louvre")),l+=w}}else o.push(Re(e,.5,.9)),o.push(E(e.x+b(a,-.15,.15)*e.w,e.z+b(a,-.15,.15)*e.d,e.w*b(a,.3,.5),e.d*b(a,.3,.5),s,s+b(a,4,8),"plant"));n&&a()<.75&&o.push(E(e.x,e.z,1.6,1.6,s,s+b(a,24,70),"clad",{mast:!0}))}function Yt(a,e,o,n){const{x:i,z:s,w:r,d:t}=e,l=[W(i,s,r,t,0,n)],d=a(),v=o>300,w=Math.max(r,t)>46,x=r>=t;if(d<.24||d<.4&&!w){const f=Math.min(r,t)*b(a,.55,.8);let c=Math.min(r-4,f*b(a,.9,1.3)),h=Math.min(t-4,f),p=n;const m=o>400?3:o>200?2:1,u=b(a,-.12,.12)*(r-c),y=b(a,-.12,.12)*(t-h);for(let z=0;z<m;z++){const k=z===m-1?o:p+(o-p)*b(a,.45,.65),M=W(i+u,s+y,c,h,p,k);l.push(M),z<m-1?(a()<.6&&(M.y1=k-b(a,4,7),l.push(E(M.x,M.z,c-.8,h-.8,M.y1,k,"louvre"))),l.push(Re({...M,y1:k},.8,.9))):ue(a,M,l,v),p=k,c*=b(a,.7,.86),h*=b(a,.7,.86)}}else if(d<.5){const f=o>380?5:o>200?4:3;let c=n;const h=o-n;for(let p=0;p<f;p++){const m=p===f-1?o-c:h/f*b(a,.8,1.2),u=Math.min(r-2,r*b(a,.5,.82)),y=Math.min(t-2,t*b(a,.5,.82)),z=b(a,-1,1)*Math.max(0,(r-u)/2-1),k=b(a,-1,1)*Math.max(0,(t-y)/2-1),M=W(i+z,s+k,u,y,c,c+m,{style:p%2?ve.OFFICE:ve.GLASS});if(l.push(M),p<f-1&&a()<.55){const I=b(a,4.5,8);M.y1-=I,l.push(E(M.x,M.z,u*.55,y*.55,M.y1,M.y1+I,a()<.5?"louvre":void 0,{style:ve.GLASS}))}l.push(Re(M,.4,.8)),p===f-1&&ue(a,M,l,v),c+=m}}else if(d<.7&&w){const f=x?r:t,c=x?t:r,h=Math.max(9,f*b(a,.14,.22)),p=(f-h)/2-2,m=Math.min(c-4,p*b(a,.8,1.2)),u=o*b(a,.78,.95),y=q=>x?[i+q*(h/2+p/2),s]:[i,s+q*(h/2+p/2)],[z,k]=y(-1),[M,I]=y(1),R=W(z,k,x?p:m,x?m:p,n,o),_=W(M,I,x?m:p,x?m:p,n,u,{style:ve.GLASS});l.push(R,_);const U=1+Math.floor(a()*3);for(let q=0;q<U;q++){const J=n+(u-n)*b(a,.3,.85),se=b(a,7,11),V=b(a,5,8);l.push(E(i,s,x?h+2:se,x?se:h+2,J,J+V,void 0,{style:ve.GLASS})),l.push(E(i,s,x?h+2.4:se+.8,x?se+.8:h+2.4,J-.8,J,"clad"))}ue(a,R,l,v),ue(a,_,l,v&&u>300)}else if(d<.85){const f=x?r-6:t-6,c=Math.min((x?t:r)-6,b(a,18,28)),h=W(i,s,x?f:c,x?c:f,n,o);l.push(h);const p=b(a,3.6,6.2),m=b(a,.7,1.4),u=Math.floor(f/p);for(let y=0;y<=u;y++){const z=-f/2+(f-u*p)/2+y*p;for(const k of[-1,1])x?l.push(E(i+z,s+k*(c/2+m/2),.45,m,n+2,o-1.5,"clad")):l.push(E(i+k*(c/2+m/2),s+z,m,.45,n+2,o-1.5,"clad"))}ue(a,h,l,v)}else{const f=Math.min(r,t)*b(a,.42,.55),c=W(i,s,f,f,n,o);l.push(c);const h=qt(a,[[1,0],[-1,0],[0,1],[0,-1]]).slice(0,2+Math.floor(a()*3));for(const[p,m]of h){const u=(p?r:t)/2-2,y=u-f/2;if(y<6)continue;const z=f*b(a,.55,.8),k=n+(o-n)*b(a,.3,.7),M=W(i+p*(f/2+y/2),s+m*(f/2+y/2),p?y:z,m?y:z,n,k,{feat:!0,style:ve.OFFICE});l.push(M,Re(M,.4,.8))}ue(a,c,l,v)}return l}function Qt(a,e,o,n,i,s,r){const t=[W(e,o,n,i,0,s)],l=a();return l<.35&&s>60?(t[0].y1=s*b(a,.55,.75),t.push(W(e+b(a,-.15,.15)*n,o+b(a,-.15,.15)*i,n*b(a,.55,.8),i*b(a,.55,.8),t[0].y1,s))):l<.5&&s>80&&t.push(W(e,o,n*.7,i*.7,s,s+b(a,8,30),{feat:!0,kind:"led"})),t}function Zt(a,e=-46){const o=a.h-e,n=Vt(a.seed,71+a.row),s=(a.row>=2?Qt(n,0,0,a.w,a.d,o):Yt(n,{x:0,z:0,w:a.w,d:a.d},o,Math.min(12,o*.12))).filter(v=>[v.x,v.z,v.w,v.d,v.y0,v.y1].every(Number.isFinite)&&v.w>.05&&v.d>.05&&v.y1-v.y0>.05),r=s.filter(v=>!v.feat),t=s.filter(v=>v.feat),l=a.row>=2?2:a.row===1?10:12,d=t.length<=l?t:[...t.slice(0,Math.floor(l/2)),...t.slice(-Math.ceil(l/2))];return[...r,...d]}function Jt(a,e=-46){return a.flatMap(o=>Zt(o,e).map(n=>({parent:o,block:n})))}const qe=720,ct=40,te=-46;function ea(a){let e=a>>>0;return()=>{e=e+1831565813>>>0;let o=e;return o=Math.imul(o^o>>>15,o|1),o^=o+Math.imul(o^o>>>7,o|61),((o^o>>>14)>>>0)/4294967296}}const ta=[{face:9.5,side:1,w:[9,18],d:[10,22],h:[22,95],row:0},{face:34,side:1,w:[12,26],d:[14,30],h:[45,170],row:1},{face:-25,side:-1,w:[9,18],d:[10,22],h:[22,95],row:0},{face:-52,side:-1,w:[12,26],d:[14,30],h:[45,170],row:1},{face:92,side:1,w:[20,34],d:[18,34],h:[80,205],row:2},{face:-112,side:-1,w:[20,34],d:[18,34],h:[80,205],row:2}];function aa(a){const e=ea(20261020),o=[],n=[];for(const i of ta){if(i.row===1&&!a.midRow||i.row===2&&!a.farRow)continue;let s=0;for(;s<qe-8;){const r=i.w[0]+e()*(i.w[1]-i.w[0]),t=i.d[0]+e()*(i.d[1]-i.d[0]),l=i.h[0]+Math.pow(e(),1.6)*(i.h[1]-i.h[0]),d=i.row===2?10+e()*14:1.5+e()*(i.row?10:5),v=i.face+i.side*(t/2+e()*(i.row?14:3)),w=-(s+r/2);if(o.push({x:v,z:w,w:t,d:r,h:l,seed:e(),row:i.row,side:i.side}),i.row===0&&a.signs&&e()<.62){const x=1+(e()<.3?1:0);for(let f=0;f<x;f++){const c=v-i.side*(t/2)-i.side*1.1;n.push({x:c,z:w+r/2-.5-f*(r-1),y:7+e()*14,cell:Math.floor(e()*16),seed:e(),side:i.side})}}s+=r+d}}return{towers:o,signs:n}}function oa(a){const e={value:0};if(!a)return{a:null,m:null,ready:e,textures:[]};const o=new gt;let n=2;const i=()=>{--n===0&&(e.value=1)},s=l=>{n=-1,console.warn("ride facade texture failed; procedural fallback retained",l)},r=o.load(new URL("./places/ride/facades_albedo.webp",document.baseURI).href,i,void 0,s),t=o.load(new URL("./places/ride/facades_mask.webp",document.baseURI).href,i,void 0,s);for(const[l,d]of[[r,Me],[t,yt]])l.colorSpace=d,l.wrapS=l.wrapT=bt,l.minFilter=St,l.magFilter=Mt,l.anisotropy=4,l.flipY=!1,l.generateMipmaps=!0;return{a:r,m:t,ready:e,textures:[r,t]}}function na(a,e,{facades:o=!0}={}){const n=new me(1,1,1);n.translate(0,.5,0);const i=new Ee;i.index=n.index,i.setAttribute("position",n.attributes.position),i.setAttribute("normal",n.attributes.normal);const s=Jt(e.towers,te),r=oa(o),t=s.length,l=new Float32Array(t*4),d=new Float32Array(t*4),v=new Float32Array(t*4),w=new Float32Array(t*4),x={clad:1,louvre:2,led:3,plant:4};s.forEach(({parent:c,block:h},p)=>{l.set([c.x,c.z,c.w,c.d],p*4),d.set([c.h,c.seed,c.row,c.side],p*4),v.set([h.x,h.z,h.y0,h.y1],p*4),w.set([h.w,h.d,x[h.kind]||0,h.feat?1:0],p*4)}),i.setAttribute("aBox",new ae(l,4)),i.setAttribute("aMeta",new ae(d,4)),i.setAttribute("aMass",new ae(v,4)),i.setAttribute("aSize",new ae(w,4)),i.instanceCount=t;const f=new A(i,$({uniforms:{...a,uFacadeA:{value:r.a},uFacadeM:{value:r.m},uFacadeReady:r.ready},vertexShader:`
        attribute vec4 aBox;
        attribute vec4 aMeta;
        attribute vec4 aMass;
        attribute vec4 aSize;
        varying vec3 vP, vN, vL;
        varying float vSeed, vH, vS, vRow, vKind, vTop, vFeat;
        void main() {
          float h = aMeta.x, seed = aMeta.y, row = aMeta.z, side = aMeta.w;
          float zc = wrapZ(aBox.y, ${qe}., ${ct}.);
          float s = uD - zc;
          float bri = onBridge(s);
          float ms = mainStreet(s);
          float x = aBox.x - side * ms * (row < .5 ? 2.5 : 10.);
          float hh = h * (1. - .25 * ms * max(0., 1. - row));
          float hs = (hh - ${te}.) / (h - ${te}.);
          float bh = aMass.w - aMass.z;
          vec3 p = position * vec3(aSize.x, bh * hs, aSize.y);
          p.y += ${te}. + aMass.z * hs;
          p.y -= bri * (hh + 120.);
          p.x += x + aMass.x;
          p.z += zc + aMass.y;
          vL = vec3(aMass.x + position.x * aSize.x, ${te}. + (aMass.z + position.y * bh) * hs, aMass.y + position.z * aSize.y);
          vN = normal;
          vSeed = floor(seed * 4096.) / 4096.;
          vH = hh;
          vTop = ${te}. + aMass.w * hs;
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
        varying float vSeed, vH, vS, vRow, vKind, vTop, vFeat;
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
          wc *= mix(.45, .2 + 1.05 * pow(hash12(cell + 3.), 3.), aa);
          vec3 c = base * (1. + pier * .35);
          // by day a lit office is a faint warm square against glass that holds the sky
          c += wc * pane * lit * step(0., v) * 1.15 * mix(1., .48, min(vRow, 1.)) * mix(.45, 1., uNight);
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
          c += vec3(.22, .08, .18) * exp(-(v - ${te}.) / 40.) * .6;
          float d = length(vP);
          c = mix(c, uFogColor, fogOf(d) * (1. - lit * pane * .15));
          // dusk air: the farther towers (and the back row) fade into a lighter haze, the near ones keep their shade
          float air = (1. - uNight) * (1. - exp(-d / 240.)) * (.5 + .25 * vRow);
          c = mix(c, vec3(.66, .53, .5), air);
          gl_FragColor = lin(c);
        }`}));return f.name="ride-towers",f.userData.textures=r.textures,f.userData.stats={parentTowers:e.towers.length,instances:t,draws:1,triangles:t*12},f}const sa=["HOTEL","BAR","24H","KTV","NOODLE","CAFE","ARCADE","NOCTILUCA","JINRUI","GOODBOAT","TAXI","BOOKS","DINER","CLUB","LIVE","OPEN"],at=["#ff4fb8","#4fe3ff","#ffb547","#9a7bff","#ff5a5a","#5affc8","#ff7ae0","#7ad0ff"];function ia(){const a=document.createElement("canvas");a.width=1024,a.height=1024;const e=a.getContext("2d");e.fillStyle="#000",e.fillRect(0,0,1024,1024),sa.forEach((n,i)=>{const s=i%8*128,r=Math.floor(i/8)*512,t=at[i%at.length];e.fillStyle="#0b0710",e.fillRect(s+10,r+10,108,492),e.strokeStyle=t,e.lineWidth=5,e.shadowColor=t,e.shadowBlur=16,e.strokeRect(s+16,r+16,96,480),e.fillStyle="#fff",e.font='bold 64px "Oxanium", "Big Shoulders Display", sans-serif',e.textAlign="center",e.textBaseline="middle";const l=n.split(""),d=Math.min(78,440/l.length);l.forEach((v,w)=>{e.shadowBlur=22,e.fillStyle=t,e.fillText(v,s+64,r+256-(l.length-1)*d/2+w*d),e.shadowBlur=0,e.fillStyle="#fff8",e.fillText(v,s+64,r+256-(l.length-1)*d/2+w*d)})});const o=new Fe(a);return o.colorSpace=Me,o.anisotropy=4,o}function ra(a,e){if(!e.signs.length)return null;const o=new xe(1.7,6.8),n=new Ee;n.index=o.index,n.setAttribute("position",o.attributes.position),n.setAttribute("uv",o.attributes.uv);const i=e.signs.length,s=new Float32Array(i*4),r=new Float32Array(i*2);e.signs.forEach((d,v)=>{s.set([d.x,d.z,d.y,d.cell],v*4),r.set([d.seed,d.side],v*2)}),n.setAttribute("aSign",new ae(s,4)),n.setAttribute("aMeta",new ae(r,2)),n.instanceCount=i;const t=ia(),l=new A(n,$({uniforms:{...a,uAtlas:{value:t}},transparent:!0,depthWrite:!1,blending:it,side:ce,vertexShader:`
        attribute vec4 aSign;
        attribute vec2 aMeta;
        varying vec2 vUv;
        varying float vFl, vFog;
        void main() {
          float zc = wrapZ(aSign.y, ${qe}., ${ct}.);
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
        }`}));return l.name="ride-signs",l.userData.textures=[t],l}function la(a){const e=new xe(2400,2400,1,1);e.rotateX(-Math.PI/2),e.translate(0,te+6,-800);const o=new A(e,$({uniforms:a,vertexShader:`
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
        }`}));return o.name="ride-ground",o.renderOrder=-8,o}const j=(a,e,o,n,i)=>a.push(...e,...o,...n,...e,...n,...i);function ft(a,e,o={}){return $({uniforms:a,side:ce,...o,vertexShader:`
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
      }`,fragmentShader:e})}function ca(a,[e,o]){const n=[],i=[],s=[];for(let x=e;x<o;x+=4){const f=n.length;j(n,[-10.3,0,0],[-10.3,6.4,0],[-10.3,6.4,-4],[-10.3,0,-4]),j(n,[5,0,0],[5,0,-4],[5,6.4,-4],[5,6.4,0]),j(n,[-10.3,6.4,0],[5,6.4,0],[5,6.4,-4],[-10.3,6.4,-4]);for(let c=f;c<n.length;c+=3)i.push(x),s.push(c-f<36?0:1);if(Math.round((x-e)/4)%2===0)for(const c of[-10.3+.25,5-.25]){const h=n.length,p=6.4-.9,m=6.4-.6,u=-.4,y=-2.8,z=c+(c<0?.18:-.18);j(n,[z,p,u],[z,m,u],[z,m,y],[z,p,y]),j(n,[c,p,u],[z,p,u],[z,p,y],[c,p,y]);for(let k=h;k<n.length;k+=3)i.push(x),s.push(2)}}for(const[x,f]of[[e,1],[o,-1]]){const c=n.length,h=0;j(n,[-34,0,h],[-10.3,0,h],[-10.3,24,h],[-34,24,h]),j(n,[5,0,h],[30,0,h],[30,24,h],[5,24,h]),j(n,[-10.3,6.4,h],[5,6.4,h],[5,24,h],[-10.3,24,h]);for(let p=c;p<n.length;p+=3)i.push(x),s.push(3);if(f>0){const p=n.length;j(n,[-6,6.4+1.2,h+.05],[2.5,6.4+1.2,h+.05],[2.5,6.4+3,h+.05],[-6,6.4+3,h+.05]);for(let m=p;m<n.length;m+=3)i.push(x),s.push(4)}}const v=new we;v.setAttribute("position",new T(n,3)),v.setAttribute("aS",new T(i,1)),v.setAttribute("aK",new T(s,1));const w=new A(v,ft(a,`
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
    }`));return w.name="ride-tunnel",w}function fa(a,[e,o]){const n=[],i=(f,c,h,p,m,u,y,z)=>{const k=new me(c-f,p-h,u-m);k.translate((f+c)/2,(h+p)/2,(m+u)/2),k.deleteAttribute("uv"),k.deleteAttribute("normal");const M=k.attributes.position.count;k.setAttribute("aS",new T(new Array(M).fill(y),1)),k.setAttribute("aK",new T(new Array(M).fill(z),1)),n.push(k)},s=e+170,r=o-190,t=92,l=-24,d=8;for(const f of[s,r]){for(const c of[l,d])i(c-1.6,c+1.6,-46,t,-1.6,1.6,f,0),i(c-.5,c+.5,t,t+1.2,-.5,.5,f,2);for(const c of[28,82])i(l,d,c,c+3,-1.2,1.2,f,0)}const v=f=>{if(f<s)return 2+(t-2)*Math.max(0,(f-e)/(s-e))**1.6;if(f>r)return 2+(t-2)*Math.max(0,(o-f)/(o-r))**1.6;const c=(f-s)/(r-s);return t-(t-10)*4*c*(1-c)};for(const f of[l+1,d-1]){for(let c=e;c<o;c+=6){const h=v(c),p=v(c+6),m=new we,u=[];j(u,[f-.35,h,0],[f+.35,h,0],[f+.35,p,-6],[f-.35,p,-6]),j(u,[f,h-.35,0],[f,h+.35,0],[f,p+.35,-6],[f,p-.35,-6]),m.setAttribute("position",new T(u,3)),m.setAttribute("aS",new T(new Array(12).fill(c),1)),m.setAttribute("aK",new T(new Array(12).fill(1),1)),n.push(m)}for(let c=e+12;c<o-6;c+=12){const h=v(c);h<3||i(f-.06,f+.06,1,h,-.06,.06,c,1)}}const w=We(n.map(f=>f.index?f.toNonIndexed():f)),x=new A(w,ft(a,`
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
    }`));return x.name="ride-bridge",x}const da=[["BAY TUNNEL","HARBOUR  2 km"],["BAY BRIDGE","NOCTILUCA BAY"],["EXIT 1 km","MAIN STREET"],["MAIN STREET","KERB  ·  CENTRE"]];function ha(){const a=document.createElement("canvas");a.width=1024,a.height=512;const e=a.getContext("2d");da.forEach(([n,i],s)=>{const r=s%2*512,t=Math.floor(s/2)*256;e.fillStyle="#0c5a4a",e.fillRect(r,t,512,256),e.strokeStyle="#e8f4ee",e.lineWidth=8,e.strokeRect(r+14,t+14,484,228),e.fillStyle="#f4faf6",e.textAlign="center",e.font='bold 64px "Oxanium", "Big Shoulders Display", sans-serif',e.fillText(n,r+256,t+110),e.font='44px "Oxanium", "Big Shoulders Display", sans-serif',e.fillText(i,r+256,t+186)});const o=new Fe(a);return o.colorSpace=Me,o.anisotropy=4,o}function va(a,e){const o=[],n=[],i=[],s=[],r=(w,x,f,c)=>{o.push(...w),n.push(...x),i.push(f),s.push(c)},t=(w,x,f,c,h,p)=>{for(const m of[h,h-.5]){const u=[[w,f,m],[x,f,m],[x,c,m],[w,f,m],[x,c,m],[w,c,m]];for(const y of u)r(y,[0,0],p,0)}};e.forEach(({s:w,board:x})=>{t(-10,-9.6,0,8.2,0,w),t(4.6,5,0,8.2,0,w),t(-10,5,7.6,8.2,0,w);const f=x%2/2,c=1-(Math.floor(x/2)+1)/2,h=-6.8,p=.6,m=4.8,u=7.5,y=.06,z=[[h,m,y,f,c],[p,m,y,f+.5,c],[p,u,y,f+.5,c+.5],[h,m,y,f,c],[p,u,y,f+.5,c+.5],[h,u,y,f,c+.5]];for(const[k,M,I,R,_]of z)r([k,M,I],[R,_],w,1)});const l=new we;l.setAttribute("position",new T(o,3)),l.setAttribute("uv",new T(n,2)),l.setAttribute("aS",new T(i,1)),l.setAttribute("aK",new T(s,1));const d=ha(),v=new A(l,$({uniforms:{...a,uBoards:{value:d}},side:ce,vertexShader:`
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
      }`}));return v.name="ride-gantries",v.userData.textures=[d],v}const dt=520,ua=40;function pa(){const a=[],e=(o,n,i,s,r,t)=>{const l=new me(o,n,i);l.translate(0,s,r),l.deleteAttribute("uv"),l.setAttribute("kind",new T(new Array(l.attributes.position.count).fill(t),1)),a.push(l)};return e(1.82,.62,4.4,.62,0,0),e(1.5,.5,2.2,1.18,.25,1),We(a)}function ma(){const a=[],e=[],o=[],n=(r,t,l,d,v,w)=>{const x=[[r,l,v],[t,l,v],[t,d,v],[r,l,v],[t,d,v],[r,d,v]],f=[[0,0],[1,0],[1,1],[0,0],[1,1],[0,1]];x.forEach((c,h)=>{a.push(...c),e.push(...f[h]),o.push(w)})},i=(r,t,l,d,v,w)=>{const x=[[r,l,d],[t,l,d],[t,l,v],[r,l,d],[t,l,v],[r,l,v]],f=[[0,0],[1,0],[1,1],[0,0],[1,1],[0,1]];x.forEach((c,h)=>{a.push(...c),e.push(...f[h]),o.push(w)})};n(-1.1,1.1,.35,.95,-2.22,0),n(-1.1,1.1,.5,1,2.22,1);for(const r of[-.7,.7]){const t=[[r,.55,-2.2],[r,.85,-2.2],[r,.85,18],[r,.55,-2.2],[r,.85,18],[r,.55,18]],l=[[0,0],[0,1],[1,1],[0,0],[1,1],[1,0]];t.forEach((d,v)=>{a.push(...d),e.push(...l[v]),o.push(2)})}i(-1.6,1.6,.03,-2.2,-14,3);const s=new we;return s.setAttribute("position",new T(a,3)),s.setAttribute("uv",new T(e,2)),s.setAttribute("kind",new T(o,1)),s}function wa(a){const e=[];let o=7;const n=()=>(o=o*16807%2147483647)/2147483647,i=Math.round(a*.4);for(let s=0;s<a;s++){const r=s>=i,t=r?be.oncoming:[be.mid,be.far];e.push({x:t[Math.floor(n()*t.length)]+(n()-.5)*.4,z0:-n()*dt,v:r?18+n()*12:16+n()*10,dir:r?1:-1,paint:n()})}return e}const ot=`
  attribute vec4 aCar; // x, z0, speed, dir (+1 oncoming, -1 our way)
  attribute float aPaint;
  attribute float kind;
  varying vec3 vP;
  varying float vKind, vPaint, vRel, vDir, vS;
  varying vec2 vUv;
  void main() {
    float dir = aCar.w;
    // oncoming cars come toward us at their own speed plus ours; our way, they move ahead of us at theirs
    float z = wrapZ(aCar.y + dir * aCar.z * uT, ${dt}., ${ua}.);
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
  }`;function nt(a,e){const o=new Ee;a.index&&(o.index=a.index);for(const s of Object.keys(a.attributes))o.setAttribute(s,a.attributes[s]);const n=new Float32Array(e.length*4),i=new Float32Array(e.length);return e.forEach((s,r)=>{n.set([s.x,s.z0,s.v,s.dir],r*4),i[r]=s.paint}),o.setAttribute("aCar",new ae(n,4)),o.setAttribute("aPaint",new ae(i,1)),o.instanceCount=e.length,o}function xa(a,e,{trails:o=!0}={}){const n=new ne;n.name="ride-traffic";const i=new A(nt(pa(),e),$({uniforms:a,vertexShader:ot,fragmentShader:`
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
        }`}));i.name="ride-traffic-bodies",n.add(i);const s=ma(),r=new A(nt(s,e),$({uniforms:{...a,uTrails:{value:o?1:0}},transparent:!0,depthWrite:!1,blending:it,side:ce,defines:{USE_UV:""},vertexShader:ot,fragmentShader:`
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
        }`}));return r.name="ride-traffic-lamps",r.renderOrder=3,n.add(r),n}const F={x:.36,z0:-.36,z1:.38,y:1.33};function ga(a){a.updateMatrixWorld(!0);const e=new rt,o=new re,n=new re;a.traverse(i=>{if(!i.isMesh||i.name)return;const s=i.material;s!=null&&s.isMeshBasicMaterial&&s.transparent&&(e.setFromObject(i).getSize(o),e.getCenter(n),o.x<.05&&o.z>2.4&&o.z<2.8&&Math.abs(n.x)>.7&&n.y>1&&n.y<1.4&&(i.name=n.x>0?"glass-right":"glass-left"))})}function ya(a,e,o){let n=null,i=1/0;const s=new re;return a.traverse(r=>{if(r.name!==e)return;r.getWorldPosition(s);const t=o?s.distanceTo(o):0;t<i&&(i=t,n=r)}),n}function ba(a,e){const o=[],n=new Map,i=new rt;return a.updateMatrixWorld(!0),a.traverse(s=>{if(!s.isMesh||s.userData.ride||s.userData.noCollide||[].concat(s.material).some(t=>t==null?void 0:t.isShaderMaterial)||(i.setFromObject(s),i.max.y<F.y||i.min.x>F.x||i.max.x<-.36||i.min.z>F.z1||i.max.z<F.z0))return;const r=t=>{if(!t||t.transparent)return t;if(!n.has(t)){const l=t.clone();l.onBeforeCompile=d=>{d.uniforms.uRoofOpen=e,d.vertexShader=d.vertexShader.replace("void main() {",`varying vec3 vRoofW;
void main() {`).replace("#include <project_vertex>",`#include <project_vertex>
  vRoofW = (modelMatrix * vec4(transformed, 1.0)).xyz;`),d.fragmentShader=d.fragmentShader.replace("void main() {",`uniform float uRoofOpen;
varying vec3 vRoofW;
void main() {
  if (uRoofOpen > 0.001 && vRoofW.y > ${F.y.toFixed(3)} && abs(vRoofW.x) < ${F.x.toFixed(3)} && vRoofW.z > ${F.z0.toFixed(3)} && vRoofW.z < ${F.z0.toFixed(3)} + ${(F.z1-F.z0).toFixed(3)} * uRoofOpen) discard;`)},l.customProgramCacheKey=()=>"ride-roof",n.set(t,l),o.push(l)}return n.get(t)};s.material=Array.isArray(s.material)?s.material.map(r):r(s.material)}),o}function Sa(){const a=document.createElement("canvas");a.width=512,a.height=224;const e=new Fe(a);e.colorSpace=Me;const o=new A(new xe(.31,.13),new Se({map:e,toneMapped:!1}));o.position.set(-.012,.983,-.9115),o.rotation.x=-.08,o.name="ride-screen";let n="";return{mesh:o,draw:s=>{const r=JSON.stringify(s);if(r===n)return;n=r;const t=a.getContext("2d");t.fillStyle="#050b14",t.fillRect(0,0,512,224),t.fillStyle="rgba(120,170,230,0.5)",t.fillRect(12,11,488,1),t.font="11px sans-serif",t.fillStyle="#7fa8d8",t.fillText(s.clock,15,25),t.fillText("MEDIA   ROUTE   CAR",190,25);const l=s.station>0;if(t.fillStyle=l?"#13243a":"#0b1420",t.fillRect(14,38,200,172),t.fillStyle=l?"#ff9ab8":"#4a6a8a",t.font="bold 15px sans-serif",t.fillText(Te[s.station].name,24,62),t.font="12px sans-serif",t.fillStyle="#9cc4ee",l){t.fillText(Te[s.station].track,24,84),t.fillText(Te[s.station].artist,24,102);for(let d=0;d<14;d++){const v=8+(d*37+s.beat*13)%40;t.fillStyle="#4aa8ff",t.fillRect(24+d*13,190-v,8,v)}}else t.fillText("F at the screen: tune in",24,84);t.strokeStyle="#2a4a6a",t.lineWidth=4,t.beginPath(),t.moveTo(250,120),t.lineTo(490,120),t.stroke(),t.strokeStyle="#4aa8ff",t.beginPath(),t.moveTo(250,120),t.lineTo(250+240*s.progress,120),t.stroke(),lt.forEach((d,v)=>{const w=250+240*d.t/X;t.fillStyle=s.chapter>=d.n?"#ff9ab8":"#2a4a6a",t.beginPath(),t.arc(w,120,6,0,Math.PI*2),t.fill(),t.fillStyle="#7fa8d8",t.font="10px sans-serif",t.fillText(String(v+1),w-3,142)}),t.fillStyle="#ffffff",t.beginPath(),t.arc(250+240*s.progress,120,8,0,Math.PI*2),t.fill(),s.done?(t.strokeStyle="#ff9ab8",t.lineWidth=3,t.beginPath(),t.moveTo(252,72),t.lineTo(259,80),t.lineTo(273,62),t.stroke(),t.fillStyle="#ff9ab8",t.font="bold 22px sans-serif",t.fillText(s.title,282,80,216)):(t.fillStyle="#9cc4ee",t.font="bold 16px sans-serif",t.fillText(s.title,250,80,248)),t.fillStyle="#9cc4ee",t.font="14px sans-serif",t.fillText(s.eta,250,178,248),e.needsUpdate=!0},tex:e}}function Ma(){const a=document.createElement("canvas");a.width=512,a.height=224;const e=new Fe(a);e.colorSpace=Me;const o=new A(new xe(.33,.13),new Se({map:e,toneMapped:!1}));o.position.set(-.39,.985,-.9115),o.rotation.x=-.08,o.name="ride-cluster";let n="";return{mesh:o,draw:(s,r)=>{const t=`${Math.round(s)}${r}`;if(t===n)return;n=t;const l=a.getContext("2d");l.setTransform(.5,0,0,.5,0,0),Gt(l,1024,448,Math.round(s),r),e.needsUpdate=!0},tex:e}}function za(){const a=new ne;a.name="ride-charm";const e=new A(new Xe(.0012,.0012,.11,4),new Se({color:10127984}));e.position.y=-.055;const o=new At;o.absarc(0,0,.032,Math.PI*.15,Math.PI*1.85,!1),o.absarc(.014,0,.026,Math.PI*1.75,Math.PI*.25,!0);const n=new A(new Tt(o,{depth:.003,bevelEnabled:!1,curveSegments:18}),new je({color:15918792,emissive:9069624,emissiveIntensity:.5,roughness:.9,side:ce}));return n.position.set(0,-.14,-.0015),n.rotation.z=.25,a.add(e,n),a.position.set(0,1.262,-.63),a}const ka=[{rough:.38,set:{roughness:.74,color:4605003}},{rough:.48,set:{roughness:.64,color:3420986}},{rough:.5,set:{roughness:.7,color:2763055}}];function Pa(a){const e=new Map;return a.traverse(o=>{const n=o.material;if(!o.isMesh||!(n!=null&&n.isMeshStandardMaterial)||n.metalness>.05)return;const i=ka.find(s=>Math.abs((n.userData.authorRough??n.roughness)-s.rough)<.015);if(i){if(!e.has(n)){const s=n.clone();n.userData.finish||(s.roughness=i.set.roughness),s.color.setHex(i.set.color),e.set(n,s)}o.material=e.get(n)}}),[...e.values()]}function Aa(){const a=new ne;a.name="ride-lights";const e=new Pt(4868704,2102296,.34),o=new Ke(16758984,1.3);o.position.set(20,30,-40),o.target.position.set(0,0,-10),o.castShadow=!0,o.shadow.mapSize.set(2048,2048),Object.assign(o.shadow.camera,{left:-30,right:30,top:30,bottom:-30,near:1,far:120}),o.shadow.bias=-4e-4,o.shadow.normalBias=.03;const n=new Ie(16761032,60,16,1.4);n.position.set(1.5,7,-20);const i=new Ie(16730832,30,9,1.6);i.position.set(-7,2,-12);const s=new Ke(10131104,.52);s.position.set(-.2,1.7,1),s.target.position.set(.1,.85,-1);const r=new Ke(16042144,0);r.position.set(-1.6,1.55,-.9),r.target.position.set(.5,.95,.2);const t=new Ie(8688832,.1,.8,2);return t.position.set(-.2,.95,-.6),a.add(e,o,o.target,n,i,s,s.target,r,r.target,t),{group:a,hemi:e,key:o,pool:n,mag:i,fill:s,win:r,disp:t}}function Ta(a,e){const o=new ne;o.name="ride-cabin";const n={value:0},i=ba(a,n),s=a.getObjectByName("sunroof");s&&(s.visible=!1);const r=new je({color:11845078,roughness:.32,metalness:.35,emissive:1316900}),t=new ne;t.name="ride-sunroof";const l=F.x*2,d=F.z1-F.z0,v=(F.z0+F.z1)/2;for(const[S,N,ee,fe]of[[l+.06,.03,0,F.z0-.015],[l+.06,.03,0,F.z1+.015],[.03,d,-.36-.015,v],[.03,d,F.x+.015,v]]){const ze=new A(new me(S,.012,N),r);ze.position.set(ee,1.392,fe),t.add(ze)}const w=new Se({color:3816026,transparent:!0,opacity:.35,depthWrite:!1,side:ce}),x=new A(new xe(l,d),w);x.rotation.x=Math.PI/2,x.position.set(0,1.455,v),o.add(t,x);const f=za();o.add(f);const c=Sa();o.add(c.mesh);const h=Ma();o.add(h.mesh);const p=new Se({color:3866522,toneMapped:!1}),m=new je({color:2764344,roughness:.45,metalness:.5}),u=new ne;u.name="ride-lightswitch",u.position.set(-.72,.79,-.858);const y=new A(new Xe(.024,.026,.022,20).rotateX(Math.PI/2),m);y.position.z=.011;const z=new A(new me(.004,.018,.004),p);z.position.set(0,.01,.023),y.add(z);const k=new A(new zt(.005,12),p);k.position.set(.04,.012,.002),u.add(y,k),o.add(u),i.push(r,w,p,m,c.mesh.material,c.tex,h.mesh.material,h.tex);for(const S of[f,t,x,c.mesh,h.mesh,u])S.traverse(N=>{N.userData.ride=!0,N.raycast=()=>{}});o.traverse(S=>{S.userData.noCollide=!0});const M=a.getObjectByName("wheel"),I=M==null?void 0:M.quaternion.clone();M&&i.push(...Pa(M));const R=ya(a,"glass-right",new re(1,1.2,0)),_=(R==null?void 0:R.position.y)??0,U=a.getObjectByName("leadcar"),q=U==null?void 0:U.position.clone(),J=new re(0,0,1),se=new kt,V={t0:-99,amp:0};return{group:o,owned:i,uniforms:{uRoofOpen:n},flick(S){V.t0=S,V.amp=.75},screen:c,update(S,N){n.value=S.roof,h.draw(S.speed*3.6,S.t>X-.5?"P":"D"),x.position.z=v+d*S.roof*.98,x.visible=S.roof>.01,R&&(R.position.y=_-.44*S.window),M&&M.quaternion.copy(I).multiply(se.setFromAxisAngle(J,O.clamp(-S.curve*900,-.5,.5)));const ee=N-V.t0,fe=V.amp*Math.exp(-ee/2.2)*Math.sin(ee*5.2);f.rotation.x=fe+S.accel*.03+Math.sin(N*1.7)*.015*Math.min(1,S.speed/10),f.rotation.z=O.clamp(S.curve*S.speed*S.speed*.5,-.4,.4)+fe*.2,p.color.setHex(S.head?3866522:1058330),y.rotation.z=S.head?-.9:0,U&&(U.position.x=q.x-1+Math.sin(N*.31)*.12,U.position.z=q.z-Math.max(0,S.t-(X-14))**2*.9,U.visible=U.position.z>-400)},dispose(){var S;for(const N of i)(S=N.dispose)==null||S.call(N)}}}const st={high:{traffic:46,rain:1400,signs:!0,midRow:!0,farRow:!0,facades:!0,trails:!0},medium:{traffic:34,rain:800,signs:!0,midRow:!0,farRow:!0,facades:!0,trails:!0},low:{traffic:24,rain:360,signs:!0,midRow:!1,farRow:!1,facades:!1,trails:!1},software:{traffic:14,rain:0,signs:!1,midRow:!1,farRow:!1,facades:!1,trails:!1}},Ae={from:"kouka",title:"Kouka",text:"Saw your car come out of the bay tunnel on the traffic feed. The water is glowing again tonight, the plankton bloom. Look left before the bridge ends."},He=a=>{try{return new URLSearchParams(location.search).get(a)}catch{return null}};function Ra(a){var o,n;const e=a.getObjectByName("leadcar");e&&e.parent!==a&&a.attach(e);for(const i of["road","wet","streaks","lights","skyDome"]){const s=a.getObjectByName(i);s&&(s.visible=!1)}a.traverse(i=>{var s;(s=i.userData)!=null&&s.update&&/mist/i.test(i.name||"")&&i.parent===a&&(i.visible=!1)});for(const i of a.children)i.isMesh&&((n=(o=i.material)==null?void 0:o.uniforms)!=null&&n.uSpeed)&&i!==e&&(i.visible=!1)}function Oa(a,{quality:e}={}){const o=he.tier||(e==="low"?"low":"high"),n=st[o]||st.high,i=He("tod")?He("tod")!=="day":Rt(),s=He("rain")!=="0",r=a.root;ga(r),Ra(r),a.walk={seated:!0,bounds:[-.3,-.3,.3,.3],eye:1.2,floorY:()=>0,colliders:[],start:{position:[0,1.2,.02],target:[0,1.06,-1]},fov:64},a.env=a.env||{},a.env.fog=i?{color:1709104,density:.0026}:{color:6178386,density:.0024},a.env.background=i?395278:3811908;const t=Bt();t.uTunnel.value.set(...L.tunnel),t.uBridge.value.set(...L.bridge),t.uRamp.value=L.ramp,t.uEnd.value=L.end,t.uNight.value=i?1:0,t.uFogColor.value.setHex(a.env.fog.color,Et),t.uFogDensity.value=a.env.fog.density;const l=new ne;l.name="ride";const d=aa({midRow:n.midRow,farRow:n.farRow,signs:n.signs}),v=[],w=g=>{g&&(l.add(g),v.push(...g.userData.textures||[]))};w($t(t)),w(la(t)),w(jt(t)),w(Xt(t)),w(na(t,d,{facades:n.facades})),w(ra(t,d)),w(ca(t,L.tunnel)),w(fa(t,L.bridge)),w(va(t,[{s:420,board:0},{s:L.tunnel[0]-170,board:0},{s:L.bridge[0]+90,board:1},{s:L.bridge[1]-260,board:2},{s:L.ramp+60,board:3}])),w(xa(t,wa(n.traffic),{trails:n.trails})),w(Kt(t,s?n.rain:0)),tt(l),r.add(l);const x={uniforms:{uWipe:{value:0},uWipeRate:{value:0}}},f=Ta(r),c=tt(It(t,x)),h=Ut();h.group.traverse(g=>{g.raycast=()=>{},g.userData.noCollide=!0});const p=Aa();r.add(f.group,c,h.group,p.group);const m=Ct(),u={t:0,chapter:1,station:0,roof:0,window:0,wipers:0,head:!0,arrived:!1,rain:0},y={roof:0,window:0},z=new Set,k=new Set,M=g=>{for(const C of k)try{C(g)}catch(B){console.error(B)}};let I=performance.now(),R=0,_=!1;const U=()=>{I=performance.now()};document.addEventListener("visibilitychange",U);const q=(g,C={})=>M({kind:"toast",text:g,...C}),J={"radio:next":()=>{m.arm();const g=m.setStation(m.stationIndex+1===Te.length?0:m.stationIndex+1);return u.station=m.stationIndex,M({kind:"radio",station:u.station}),g.url?`${g.name} · "${g.track}", ${g.artist}`:"The radio clicks off. Just the road now."},"radio:off":()=>(m.arm(),m.setStation(0),u.station=0,M({kind:"radio",station:0}),"The radio clicks off. Just the road now."),"sunroof:toggle":()=>(m.arm(),m.click(),y.roof=y.roof>.5?0:1,y.roof?"The sunroof slides back. Cold air, and the towers leaning in overhead.":"The sunroof slides shut. The city sound drops away."),"window:toggle":()=>(m.arm(),m.click(),y.window=y.window>.5?0:1,y.window?"The window winds down. The wind comes in loud, smelling of rain and the sea.":"The window winds up. Quiet again."),"wipers:cycle":()=>(m.arm(),m.click("wiper"),u.wipers=(u.wipers+1)%3,u.wipers===0&&(_=!0),["Wipers off.","Wipers on.","Wipers fast."][u.wipers]+(u.rain<.05&&u.wipers?" The glass is dry; they squeak once and keep time.":"")),"lights:toggle":()=>(m.arm(),m.click(u.head?"off":"switch"),u.head=!u.head,u.head?"Headlamps on. The lane comes back out of the dark.":"Headlamps off. Between the street lamps the road goes dark; the car does not mind."),"charm:flick":()=>(f.flick(se()),"You flick the paper moon. It spins on its thread, and the dashboard light runs around its edge."),"route:show":()=>{const g=et(u.t),C=Math.max(0,Math.round(X-u.t));return u.arrived?"Main street, at the kerb. The route screen says: you have arrived.":`Chapter ${g.n} of 3, ${g.title}. Main street in ${Math.floor(C/60)}:${String(C%60).padStart(2,"0")}.`},"ride:restart":()=>(V.seek(0),z.clear(),"The car pulls away from the kerb again."),horn:()=>(m.arm(),m.click(),null)},se=()=>performance.now()/1e3,V={get state(){return{...u,length:_e,end:X}},message:Ae,subscribe(g){return k.add(g),()=>k.delete(g)},use(g){var K,D,Q,P;const C=(Q=(D=(K=g==null?void 0:g.interactable)==null?void 0:K.primary)==null?void 0:D.effect)==null?void 0:Q.device;if(!C||!J[C])return!1;const Y=J[C]()||((P=g.lines)==null?void 0:P[0])||null;return Y&&q(Y,{device:C}),!0},seek(g){u.t=Math.max(0,Math.min(_e,g)),I=performance.now(),g<Ue.mid&&z.delete("mid"),g<X&&(z.delete("arrive"),u.arrived=!1)},audio:m};a.ride=V;const S={pos:new re,rx:0,ry:0,rz:0,on:!1,p0:new re,r0:new Ft},N=(g,C,B,Y)=>{const K=performance.now(),D=Math.min(1,(K-I)/1e3);I=K;const Q=document.hidden||!!he.hold||!!he.settling;Q||(u.t=Math.min(_e,u.t+D));const P=u.t,H=Dt(P),Z=Ge(P),le=Ot(P),De=(Ge(P+.25)-Ge(P-.25))/.5;t.uT.value=K/1e3%3600,t.uD.value=H,t.uSpeed.value=Z,t.uCurve.value=le,t.uHill.value=Lt(P),t.uHeading.value=Nt(P),u.rain=s?O.clamp((P-Ue.rain)/10,0,1)*(P>X+2?.8:1):0,t.uRain.value=u.rain,t.uHead.value=O.lerp(t.uHead.value,u.head?1:0,Math.min(1,D*6)),u.roof+=O.clamp(y.roof-u.roof,-D*.7,D*.7),u.window+=O.clamp(y.window-u.window,-D*.6,D*.6);const Oe=[0,.75,1.35][u.wipers];if(Oe)R+=D*Oe,_=!1;else if(_){const ie=Math.ceil(R);R=Math.min(ie,R+D*.75),R>=ie&&(_=!1)}x.uniforms.uWipe.value=R,x.uniforms.uWipeRate.value=Oe||(_?.75:0);const Le=R%1,ht=Le<.5?Le*2:2-Le*2;for(const ie of h.blades)ie.rotation.z=ht*_t;f.update({t:P,roof:u.roof,window:u.window,curve:le,speed:Z,accel:De,head:u.head},K/1e3);const de=et(P),Ne=Math.max(0,Math.round(X-P)),ke=u.arrived||Ne===0;f.screen.draw({station:u.station,beat:u.station?Math.floor(K/500):0,chapter:de.n,done:ke,title:ke?"Arrived":`${de.n} · ${de.title}`,progress:ke?1:Math.min(1,P/X),eta:ke?"Main street, at the kerb":`Main street ${Math.floor(Ne/60)}:${String(Ne%60).padStart(2,"0")}`,clock:new Date().toTimeString().slice(0,5)});const G=H>L.tunnel[0]&&H<L.tunnel[1]?1:0,ge=H>L.bridge[0]+40&&H<L.bridge[1]-40?1:0,vt=H%oe/oe,Pe=Math.exp(-(((vt-.5)*oe/7)**2)),Ve=Math.exp(-(((H%8-4)/2.2)**2)),{key:Ye,hemi:Be,pool:ut,mag:Qe,fill:Ze,win:pt}=p;Ye.intensity=G?.6+.9*Ve:(ge?.95:1.1)+.35*Pe,Ye.color.setHex(G?16754768:ge?12110079:i?16758984:16765620),Be.intensity=G?.2+.35*Ve:i?ge?.42:.3+.22*Pe:.5+.1*Pe,Be.color.setHex(G?16752736:i?4999256:8288392),pt.intensity=i||G?0:.75,Be.groundColor.setHex(ge?932410:G?3810320:2102296),ut.intensity=(G?25:60)*(u.head?1:.35);const mt=Math.floor(H/18)%3;Qe.color.setHex(G?16748608:ge?4247776:[16730832,4905215,16742970][mt]),Qe.intensity=G?10:30;const wt=(i?.52:.85)*(G?.85:he.still?1:.94+.16*Pe);if(Ze.intensity=O.lerp(Ze.intensity,wt,Math.min(1,D*8)),m.update({speed:Z,wind:Math.max(u.window,u.roof*.6),rain:u.rain,tunnel:G,paused:Q}),Y!=null&&Y.warp&&fe(Y.warp,{v:Z,accel:De,tun:G,D:H,camera:B}),(de.n!==u.chapter||!z.has("chapter1"))&&(u.chapter=de.n,z.add("chapter1"),M({kind:"chapter",chapter:de})),!z.has("start")&&P>2.5&&(z.add("start"),$e("ride:start",{place:"car-1",chapters:lt.length,length:X})),!z.has("mid")&&P>=Ue.mid&&(z.add("mid"),$e("ride:mid",{place:"car-1",where:"bay-bridge",hie:Ae.from,message:Ae.text}),M({kind:"message",message:Ae})),!z.has("arrive")&&P>=X&&(z.add("arrive"),u.arrived=!0,$e("ride:arrive",{place:"car-1",to:"street-1"}),M({kind:"arrive"})),B&&!he.still){const ie=Math.min(1,Z/20);S.pos.set(Math.sin(K*.0021)*.002*ie,(Math.sin(K*.0083)*.0022+Math.sin(K*.0191)*.0011)*ie+(ze(K)-.5)*6e-4*ie,0),S.rz=O.clamp(-le*Z*Z*.035,-.035,.035),S.ry=O.clamp(le*18,-.03,.03),S.rx=O.clamp(-De*.004,-.02,.02),S.p0.copy(B.position),S.r0.copy(B.rotation),B.position.add(S.pos),B.rotation.x+=S.rx,B.rotation.y+=S.ry,B.rotation.z+=S.rz,S.on=!0}},ee=new re,fe=(g,{v:C,accel:B,tun:Y,D:K,camera:D})=>{const Q=he.still||o==="software"||!D,P=O.smoothstep(C,17,25),H=O.clamp(B/3,0,1),Z=K-L.tunnel[1];if(g.amount=Q?0:Math.min(.85,.5*P+.3*H+.25*Y),g.lines=Q?0:Math.min(.6,O.smoothstep(B,.3,1)*P+.35*Y),g.flash=Q||Z<0?0:.8*Math.exp(-Z/10),g.near=2.5,!Q){const le=O.smoothstep(-D.getWorldDirection(ee).z,.35,.75);g.amount*=le,g.lines*=le,ee.set(0,1.2,-100).project(D),g.center[0]=O.clamp(ee.x*.5+.5,0,1),g.center[1]=O.clamp(ee.y*.5+.5,0,1)}},ze=g=>{const C=Math.sin(g*12.9898)*43758.5453;return C-Math.floor(C)};a.update=N,a.afterRender=g=>{S.on&&(g.position.copy(S.p0),g.rotation.copy(S.r0),S.on=!1)};const Ce=a.dispose;return a.dispose=()=>{Ce==null||Ce(),document.removeEventListener("visibilitychange",U),m.dispose(),f.dispose(),k.clear(),l.traverse(g=>{g.isMesh&&(g.geometry.dispose(),g.material.dispose())});for(const g of v)g.dispose();c.geometry.dispose(),c.material.dispose(),h.material.dispose()},V}export{Oa as attachRide};
