import{aN as dr,J as cr,aU as O,V as v,a8 as S,aa as ie,G as ve,x as Jt,t as Kt,d as en,aO as qe,aR as wt,bs as hr,S as pr,bt as fr,bu as mr,bv as xe,ap as $,C as H,i as J,ae as ge,L as F,bw as _r,K as ee,aS as gr,aQ as vr,bx as tn,aV as Tt,by as Mr,bz as We,h as me,aG as oi,a_ as si,aW as xr,b1 as yr,b2 as wr,bA as Tr,j as E,r as U,bB as Rr,bC as Sr,bD as Ar,bE as Er,bF as nn,bG as Pr,bH as Lr,bI as br,p as Ir,bJ as rn,at as ue,au as Cr,bK as Nr,bL as Or,b as Ur,bM as Vr,bN as kr,bO as Dr}from"./index-DjKq0-VJ.js";import{C as Br,u as Fr,a as Hr}from"./react-three-fiber.esm-0q7lHoYZ.js";import{G as Wr}from"./GLTFLoader-C9diDS-S.js";const Rt=["happy","angry","sad","relaxed","surprised"],Pe=["aa","ih","ou","ee","oh"],zr=Object.freeze(Object.fromEntries(Rt.map(e=>[e,0]))),ai=Object.freeze({default:{face:{},head:[0,0,0],gaze:[0,0]},happy:{face:{happy:.8},head:[-.03,0,.06],gaze:[0,0]},peaceful:{face:{relaxed:.6},head:[.03,0,.05],gaze:[0,-.02]},shy:{face:{happy:.32,relaxed:.25},head:[.13,.12,.09],gaze:[.22,-.2]},depressing:{face:{sad:.8},head:[.15,0,-.03],gaze:[0,-.28]},angry:{face:{angry:.85},head:[.07,0,0],gaze:[0,0]},surprised:{face:{surprised:.85},head:[-.07,0,0],gaze:[0,.03]},thinking:{face:{angry:.2,relaxed:.15},head:[-.05,.1,-.1],gaze:[-.24,.2]}}),jr={sad:"depressing",sorrow:"depressing",relaxed:"peaceful",calm:"peaceful",joy:"happy",neutral:"default"};function li(e){const t=String(e||"").toLowerCase().trim();return ai[t]?t:jr[t]||"default"}function Gr(e){const t=ai[li(e)];return{face:{...zr,...t.face},head:[...t.head],gaze:[...t.gaze]}}function on(e,t,n,i=8){const r=1-Math.exp(-i*Math.max(0,n)),o={};for(const a of new Set([...Object.keys(e),...Object.keys(t)])){const u=e[a]||0;o[a]=u+((t[a]||0)-u)*r}return o}const sn=(e,t,n,i=6)=>{const r=1-Math.exp(-i*Math.max(0,n));return t.map((o,a)=>(e[a]||0)+(o-(e[a]||0))*r)},Xr={happy:.55,surprised:.45,angry:.8,sad:.8,relaxed:.7};function qr(e,t){if(!t)return e;const n={};for(const[i,r]of Object.entries(e))n[i]=r*(Xr[i]??1);return n}const $e=150;function Yr(e=Math.random){return 2e3+e()*4e3}function Qr(e){if(e<0||e>=$e)return 0;const t=$e*.4;return e<t?e/t:1-(e-t)/($e-t)}const $r=14,an={",":.18,";":.2,":":.2,".":.36,"!":.36,"?":.36,"…":.4},Ze={a:"aa",i:"ih",y:"ih",u:"ou",w:"ou",e:"ee",o:"oh"},Zr={oo:"ou",ee:"ih",ea:"ee",ou:"ou",ow:"oh",ai:"ee",ay:"ee",ie:"ih",oa:"oh",au:"oh"},Jr=new Set(["m","b","p"]),Kr=/[぀-ヿ㐀-鿿가-힯]/;function eo(e=""){return String(e).replace(/\([^()]*\)|（[^（）]*）|\*[^*\n]+\*/g," ").replace(/\s+[—–]\s+/g,", ").replace(/\s+/g," ").trim()}function to(e,{cps:t=$r}={}){const n=String(e||"").toLowerCase(),i=1/t,r=[];let o=0;const a=(s,l)=>r.push({...s,n:l});let u="aa";for(let s=0;s<n.length;s++){const l=n[s];if(an[l]!==void 0){a({t:o,v:null,w:0},s+1),o+=an[l];continue}if(/\s/.test(l)){a({t:o,v:null,w:0},s+1),o+=i*.6;continue}if(Kr.test(l)){a({t:o,v:Pe[l.codePointAt(0)%Pe.length],w:.75},s+1),a({t:o+i*2.2,v:null,w:.1},s+1),o+=i*3;continue}const d=Zr[n.slice(s,s+2)];if(d){a({t:o,v:d,w:.95},s+2),u=d,o+=i*2,s++;continue}if(Ze[l]){a({t:o,v:Ze[l],w:l==="y"||l==="w"?.55:.95},s+1),u=Ze[l],o+=i;continue}if(Jr.has(l)){a({t:o,v:null,w:0},s+1),o+=i;continue}if(l==="f"||l==="v"){a({t:o,v:"ih",w:.18},s+1),o+=i;continue}if(/[a-z0-9]/.test(l)){a({t:o,v:u,w:.35},s+1),o+=i;continue}}return r.push({t:o,v:null,w:0,n:n.length}),{duration:o,keys:r,length:n.length}}function no(e,t){const n=Object.fromEntries(Pe.map(u=>[u,0]));if(!e||t<0||t>=e.duration)return n;const{keys:i}=e;let r=0,o=i.length-1;for(;r<o;){const u=r+o+1>>1;i[u].t<=t?r=u:o=u-1}const a=i[r];return a.v&&(n[a.v]=a.w),n}function io(e,t){if(!e)return 0;if(t>=e.duration)return e.length;let n=0;for(const i of e.keys){if(i.t>t)break;n=i.n}return n}/*!
 * @pixiv/three-vrm v3.5.5
 * VRM file loader for three.js.
 *
 * Copyright (c) 2019-2026 pixiv Inc.
 * @pixiv/three-vrm is distributed under MIT License
 * https://github.com/pixiv/three-vrm/blob/release/LICENSE
 */var Ue=(e,t,n)=>new Promise((i,r)=>{var o=s=>{try{u(n.next(s))}catch(l){r(l)}},a=s=>{try{u(n.throw(s))}catch(l){r(l)}},u=s=>s.done?i(s.value):Promise.resolve(s.value).then(o,a);u((n=n.apply(e,t)).next())}),I=(e,t,n)=>new Promise((i,r)=>{var o=s=>{try{u(n.next(s))}catch(l){r(l)}},a=s=>{try{u(n.throw(s))}catch(l){r(l)}},u=s=>s.done?i(s.value):Promise.resolve(s.value).then(o,a);u((n=n.apply(e,t)).next())}),ln=class extends ge{constructor(e){super(),this.weight=0,this.isBinary=!1,this.overrideBlink="none",this.overrideLookAt="none",this.overrideMouth="none",this._binds=[],this.name=`VRMExpression_${e}`,this.expressionName=e,this.type="VRMExpression",this.visible=!1}get binds(){return this._binds}get overrideBlinkAmount(){return this.overrideBlink==="block"?0<this.outputWeight?1:0:this.overrideBlink==="blend"?this.outputWeight:0}get overrideLookAtAmount(){return this.overrideLookAt==="block"?0<this.outputWeight?1:0:this.overrideLookAt==="blend"?this.outputWeight:0}get overrideMouthAmount(){return this.overrideMouth==="block"?0<this.outputWeight?1:0:this.overrideMouth==="blend"?this.outputWeight:0}get outputWeight(){return this.isBinary?this.weight>.5?1:0:this.weight}addBind(e){this._binds.push(e)}deleteBind(e){const t=this._binds.indexOf(e);t>=0&&this._binds.splice(t,1)}applyWeight(e){var t;let n=this.outputWeight;n*=(t=e==null?void 0:e.multiplier)!=null?t:1,this.isBinary&&n<1&&(n=0),this._binds.forEach(i=>i.applyWeight(n))}clearAppliedWeight(){this._binds.forEach(e=>e.clearAppliedWeight())}};function ui(e,t,n){var i,r;const o=e.parser.json,a=(i=o.nodes)==null?void 0:i[t];if(a==null)return console.warn(`extractPrimitivesInternal: Attempt to use nodes[${t}] of glTF but the node doesn't exist`),null;const u=a.mesh;if(u==null)return null;const s=(r=o.meshes)==null?void 0:r[u];if(s==null)return console.warn(`extractPrimitivesInternal: Attempt to use meshes[${u}] of glTF but the mesh doesn't exist`),null;const l=s.primitives.length,d=[];return n.traverse(c=>{d.length<l&&c.isMesh&&d.push(c)}),d}function un(e,t){return I(this,null,function*(){const n=yield e.parser.getDependency("node",t);return ui(e,t,n)})}function dn(e){return I(this,null,function*(){const t=yield e.parser.getDependencies("node"),n=new Map;return t.forEach((i,r)=>{const o=ui(e,r,i);o!=null&&n.set(r,o)}),n})}var mt={Aa:"aa",Ih:"ih",Ou:"ou",Ee:"ee",Oh:"oh",Blink:"blink",Happy:"happy",Angry:"angry",Sad:"sad",Relaxed:"relaxed",LookUp:"lookUp",Surprised:"surprised",LookDown:"lookDown",LookLeft:"lookLeft",LookRight:"lookRight",BlinkLeft:"blinkLeft",BlinkRight:"blinkRight",Neutral:"neutral"};function di(e){return Math.max(Math.min(e,1),0)}var cn=class ci{constructor(){this.blinkExpressionNames=["blink","blinkLeft","blinkRight"],this.lookAtExpressionNames=["lookLeft","lookRight","lookUp","lookDown"],this.mouthExpressionNames=["aa","ee","ih","oh","ou"],this._expressions=[],this._expressionMap={}}get expressions(){return this._expressions.concat()}get expressionMap(){return Object.assign({},this._expressionMap)}get presetExpressionMap(){const t={},n=new Set(Object.values(mt));return Object.entries(this._expressionMap).forEach(([i,r])=>{n.has(i)&&(t[i]=r)}),t}get customExpressionMap(){const t={},n=new Set(Object.values(mt));return Object.entries(this._expressionMap).forEach(([i,r])=>{n.has(i)||(t[i]=r)}),t}copy(t){return this._expressions.concat().forEach(i=>{this.unregisterExpression(i)}),t._expressions.forEach(i=>{this.registerExpression(i)}),this.blinkExpressionNames=t.blinkExpressionNames.concat(),this.lookAtExpressionNames=t.lookAtExpressionNames.concat(),this.mouthExpressionNames=t.mouthExpressionNames.concat(),this}clone(){return new ci().copy(this)}getExpression(t){var n;return(n=this._expressionMap[t])!=null?n:null}registerExpression(t){this._expressions.push(t),this._expressionMap[t.expressionName]=t}unregisterExpression(t){const n=this._expressions.indexOf(t);n===-1&&console.warn("VRMExpressionManager: The specified expressions is not registered"),this._expressions.splice(n,1),delete this._expressionMap[t.expressionName]}getValue(t){var n;const i=this.getExpression(t);return(n=i==null?void 0:i.weight)!=null?n:null}setValue(t,n){const i=this.getExpression(t);i&&(i.weight=di(n))}resetValues(){this._expressions.forEach(t=>{t.weight=0})}getExpressionTrackName(t){const n=this.getExpression(t);return n?`${n.name}.weight`:null}update(){const t=this._calculateWeightMultipliers();this._expressions.forEach(n=>{n.clearAppliedWeight()}),this._expressions.forEach(n=>{let i=1;const r=n.expressionName;this.blinkExpressionNames.indexOf(r)!==-1&&(i*=t.blink),this.lookAtExpressionNames.indexOf(r)!==-1&&(i*=t.lookAt),this.mouthExpressionNames.indexOf(r)!==-1&&(i*=t.mouth),n.applyWeight({multiplier:i})})}_calculateWeightMultipliers(){let t=1,n=1,i=1;return this._expressions.forEach(r=>{t-=r.overrideBlinkAmount,n-=r.overrideLookAtAmount,i-=r.overrideMouthAmount}),t=Math.max(0,t),n=Math.max(0,n),i=Math.max(0,i),{blink:t,lookAt:n,mouth:i}}},ye={Color:"color",EmissionColor:"emissionColor",ShadeColor:"shadeColor",RimColor:"rimColor",OutlineColor:"outlineColor"},ro={_Color:ye.Color,_EmissionColor:ye.EmissionColor,_ShadeColor:ye.ShadeColor,_RimColor:ye.RimColor,_OutlineColor:ye.OutlineColor},oo=new H,hi=class pi{constructor({material:t,type:n,targetValue:i,targetAlpha:r}){this.material=t,this.type=n,this.targetValue=i,this.targetAlpha=r??1;const o=this._initColorBindState(),a=this._initAlphaBindState();this._state={color:o,alpha:a}}applyWeight(t){const{color:n,alpha:i}=this._state;if(n!=null){const{propertyName:r,deltaValue:o}=n,a=this.material[r];a!=null&&a.add(oo.copy(o).multiplyScalar(t))}if(i!=null){const{propertyName:r,deltaValue:o}=i;this.material[r]!=null&&(this.material[r]+=o*t)}}clearAppliedWeight(){const{color:t,alpha:n}=this._state;if(t!=null){const{propertyName:i,initialValue:r}=t,o=this.material[i];o!=null&&o.copy(r)}if(n!=null){const{propertyName:i,initialValue:r}=n;this.material[i]!=null&&(this.material[i]=r)}}_initColorBindState(){var t,n,i;const{material:r,type:o,targetValue:a}=this,u=this._getPropertyNameMap(),s=(n=(t=u==null?void 0:u[o])==null?void 0:t[0])!=null?n:null;if(s==null)return console.warn(`Tried to add a material color bind to the material ${(i=r.name)!=null?i:"(no name)"}, the type ${o} but the material or the type is not supported.`),null;const d=r[s].clone(),c=new H(a.r-d.r,a.g-d.g,a.b-d.b);return{propertyName:s,initialValue:d,deltaValue:c}}_initAlphaBindState(){var t,n,i;const{material:r,type:o,targetAlpha:a}=this,u=this._getPropertyNameMap(),s=(n=(t=u==null?void 0:u[o])==null?void 0:t[1])!=null?n:null;if(s==null&&a!==1)return console.warn(`Tried to add a material alpha bind to the material ${(i=r.name)!=null?i:"(no name)"}, the type ${o} but the material or the type does not support alpha.`),null;if(s==null)return null;const l=r[s],d=a-l;return{propertyName:s,initialValue:l,deltaValue:d}}_getPropertyNameMap(){var t,n;return(n=(t=Object.entries(pi._propertyNameMapMap).find(([i])=>this.material[i]===!0))==null?void 0:t[1])!=null?n:null}};hi._propertyNameMapMap={isMeshStandardMaterial:{color:["color","opacity"],emissionColor:["emissive",null]},isMeshBasicMaterial:{color:["color","opacity"]},isMToonMaterial:{color:["color","opacity"],emissionColor:["emissive",null],outlineColor:["outlineColorFactor",null],matcapColor:["matcapFactor",null],rimColor:["parametricRimColorFactor",null],shadeColor:["shadeColorFactor",null]}};var hn=hi,ze=class{constructor({primitives:e,index:t,weight:n}){this.primitives=e,this.index=t,this.weight=n}applyWeight(e){this.primitives.forEach(t=>{var n;((n=t.morphTargetInfluences)==null?void 0:n[this.index])!=null&&(t.morphTargetInfluences[this.index]+=this.weight*e)})}clearAppliedWeight(){this.primitives.forEach(e=>{var t;((t=e.morphTargetInfluences)==null?void 0:t[this.index])!=null&&(e.morphTargetInfluences[this.index]=0)})}},pn=new me,fi=class mi{constructor({material:t,scale:n,offset:i}){var r,o;this.material=t,this.scale=n,this.offset=i;const a=(r=Object.entries(mi._propertyNamesMap).find(([u])=>t[u]===!0))==null?void 0:r[1];a==null?(console.warn(`Tried to add a texture transform bind to the material ${(o=t.name)!=null?o:"(no name)"} but the material is not supported.`),this._properties=[]):(this._properties=[],a.forEach(u=>{var s;const l=(s=t[u])==null?void 0:s.clone();if(!l)return null;t[u]=l;const d=l.offset.clone(),c=l.repeat.clone(),p=i.clone().sub(d),h=n.clone().sub(c);this._properties.push({name:u,initialOffset:d,deltaOffset:p,initialScale:c,deltaScale:h})}))}applyWeight(t){this._properties.forEach(n=>{const i=this.material[n.name];i!==void 0&&(i.offset.add(pn.copy(n.deltaOffset).multiplyScalar(t)),i.repeat.add(pn.copy(n.deltaScale).multiplyScalar(t)))})}clearAppliedWeight(){this._properties.forEach(t=>{const n=this.material[t.name];n!==void 0&&(n.offset.copy(t.initialOffset),n.repeat.copy(t.initialScale))})}};fi._propertyNamesMap={isMeshStandardMaterial:["map","emissiveMap","bumpMap","normalMap","displacementMap","roughnessMap","metalnessMap","alphaMap"],isMeshBasicMaterial:["map","specularMap","alphaMap"],isMToonMaterial:["map","normalMap","emissiveMap","shadeMultiplyTexture","rimMultiplyTexture","outlineWidthMultiplyTexture","uvAnimationMaskTexture"]};var fn=fi,so=new Set(["1.0","1.0-beta"]),_i=class gi{get name(){return"VRMExpressionLoaderPlugin"}constructor(t){this.parser=t}afterRoot(t){return I(this,null,function*(){t.userData.vrmExpressionManager=yield this._import(t)})}_import(t){return I(this,null,function*(){const n=yield this._v1Import(t);if(n)return n;const i=yield this._v0Import(t);return i||null})}_v1Import(t){return I(this,null,function*(){var n,i;const r=this.parser.json;if(!(((n=r.extensionsUsed)==null?void 0:n.indexOf("VRMC_vrm"))!==-1))return null;const a=(i=r.extensions)==null?void 0:i.VRMC_vrm;if(!a)return null;const u=a.specVersion;if(!so.has(u))return console.warn(`VRMExpressionLoaderPlugin: Unknown VRMC_vrm specVersion "${u}"`),null;const s=a.expressions;if(!s)return null;const l=new Set(Object.values(mt)),d=new Map;s.preset!=null&&Object.entries(s.preset).forEach(([p,h])=>{if(h!=null){if(!l.has(p)){console.warn(`VRMExpressionLoaderPlugin: Unknown preset name "${p}" detected. Ignoring the expression`);return}d.set(p,h)}}),s.custom!=null&&Object.entries(s.custom).forEach(([p,h])=>{if(l.has(p)){console.warn(`VRMExpressionLoaderPlugin: Custom expression cannot have preset name "${p}". Ignoring the expression`);return}d.set(p,h)});const c=new cn;return yield Promise.all(Array.from(d.entries()).map(p=>I(this,[p],function*([h,_]){var g,m,f,x,T,y,w;const R=new ln(h);if(t.scene.add(R),R.isBinary=(g=_.isBinary)!=null?g:!1,R.overrideBlink=(m=_.overrideBlink)!=null?m:"none",R.overrideLookAt=(f=_.overrideLookAt)!=null?f:"none",R.overrideMouth=(x=_.overrideMouth)!=null?x:"none",(T=_.morphTargetBinds)==null||T.forEach(A=>I(this,null,function*(){var P;if(A.node===void 0||A.index===void 0)return;const N=yield un(t,A.node),b=A.index;if(!N.every(C=>Array.isArray(C.morphTargetInfluences)&&b<C.morphTargetInfluences.length)){console.warn(`VRMExpressionLoaderPlugin: ${_.name} attempts to index morph #${b} but not found.`);return}R.addBind(new ze({primitives:N,index:b,weight:(P=A.weight)!=null?P:1}))})),_.materialColorBinds||_.textureTransformBinds){const A=[];t.scene.traverse(P=>{const N=P.material;N&&(Array.isArray(N)?A.push(...N):A.push(N))}),(y=_.materialColorBinds)==null||y.forEach(P=>I(this,null,function*(){A.filter(b=>{var C;const D=(C=this.parser.associations.get(b))==null?void 0:C.materials;return P.material===D}).forEach(b=>{R.addBind(new hn({material:b,type:P.type,targetValue:new H().fromArray(P.targetValue),targetAlpha:P.targetValue[3]}))})})),(w=_.textureTransformBinds)==null||w.forEach(P=>I(this,null,function*(){A.filter(b=>{var C;const D=(C=this.parser.associations.get(b))==null?void 0:C.materials;return P.material===D}).forEach(b=>{var C,D;R.addBind(new fn({material:b,offset:new me().fromArray((C=P.offset)!=null?C:[0,0]),scale:new me().fromArray((D=P.scale)!=null?D:[1,1])}))})}))}c.registerExpression(R)}))),c})}_v0Import(t){return I(this,null,function*(){var n;const i=this.parser.json,r=(n=i.extensions)==null?void 0:n.VRM;if(!r)return null;const o=r.blendShapeMaster;if(!o)return null;const a=new cn,u=o.blendShapeGroups;if(!u)return a;const s=new Set;return yield Promise.all(u.map(l=>I(this,null,function*(){var d;const c=l.presetName,p=c!=null&&gi.v0v1PresetNameMap[c]||null,h=p??l.name;if(h==null){console.warn("VRMExpressionLoaderPlugin: One of custom expressions has no name. Ignoring the expression");return}if(s.has(h)){console.warn(`VRMExpressionLoaderPlugin: An expression preset ${c} has duplicated entries. Ignoring the expression`);return}s.add(h);const _=new ln(h);t.scene.add(_),_.isBinary=(d=l.isBinary)!=null?d:!1,l.binds&&l.binds.forEach(m=>I(this,null,function*(){var f;if(m.mesh===void 0||m.index===void 0)return;const x=[];if((f=i.nodes)==null||f.forEach((y,w)=>{y.mesh===m.mesh&&x.push(w)}),x.length===0){console.warn(`VRMExpressionLoaderPlugin: ${l.name} attempts to bind a morph target to the mesh #${m.mesh} but the mesh is not found or not used in the scene. Ignoring the bind.`);return}const T=m.index;yield Promise.all(x.map(y=>I(this,null,function*(){var w;const R=yield un(t,y);if(!R.every(A=>Array.isArray(A.morphTargetInfluences)&&T<A.morphTargetInfluences.length)){console.warn(`VRMExpressionLoaderPlugin: ${l.name} attempts to index ${T}th morph but not found.`);return}_.addBind(new ze({primitives:R,index:T,weight:.01*((w=m.weight)!=null?w:100)}))})))}));const g=l.materialValues;g&&g.length!==0&&g.forEach(m=>{if(m.materialName===void 0||m.propertyName===void 0||m.targetValue===void 0)return;const f=[];t.scene.traverse(T=>{if(T.material){const y=T.material;Array.isArray(y)?f.push(...y.filter(w=>(w.name===m.materialName||w.name===m.materialName+" (Outline)")&&f.indexOf(w)===-1)):y.name===m.materialName&&f.indexOf(y)===-1&&f.push(y)}});const x=m.propertyName;f.forEach(T=>{if(x==="_MainTex_ST"){const w=new me(m.targetValue[0],m.targetValue[1]),R=new me(m.targetValue[2],m.targetValue[3]);R.y=1-R.y-w.y,_.addBind(new fn({material:T,scale:w,offset:R}));return}const y=ro[x];if(y){_.addBind(new hn({material:T,type:y,targetValue:new H().fromArray(m.targetValue),targetAlpha:m.targetValue[3]}));return}console.warn(x+" is not supported")})}),a.registerExpression(_)}))),a})}};_i.v0v1PresetNameMap={a:"aa",e:"ee",i:"ih",o:"oh",u:"ou",blink:"blink",joy:"happy",angry:"angry",sorrow:"sad",fun:"relaxed",lookup:"lookUp",lookdown:"lookDown",lookleft:"lookLeft",lookright:"lookRight",blink_l:"blinkLeft",blink_r:"blinkRight",neutral:"neutral"};var ao=_i,St=class pe{constructor(t,n){this._firstPersonOnlyLayer=pe.DEFAULT_FIRSTPERSON_ONLY_LAYER,this._thirdPersonOnlyLayer=pe.DEFAULT_THIRDPERSON_ONLY_LAYER,this._initializedLayers=!1,this.humanoid=t,this.meshAnnotations=n}copy(t){if(this.humanoid!==t.humanoid)throw new Error("VRMFirstPerson: humanoid must be same in order to copy");return this.meshAnnotations=t.meshAnnotations.map(n=>({meshes:n.meshes.concat(),type:n.type})),this}clone(){return new pe(this.humanoid,this.meshAnnotations).copy(this)}get firstPersonOnlyLayer(){return this._firstPersonOnlyLayer}get thirdPersonOnlyLayer(){return this._thirdPersonOnlyLayer}setup({firstPersonOnlyLayer:t=pe.DEFAULT_FIRSTPERSON_ONLY_LAYER,thirdPersonOnlyLayer:n=pe.DEFAULT_THIRDPERSON_ONLY_LAYER}={}){this._initializedLayers||(this._firstPersonOnlyLayer=t,this._thirdPersonOnlyLayer=n,this.meshAnnotations.forEach(i=>{i.meshes.forEach(r=>{i.type==="firstPersonOnly"?(r.layers.set(this._firstPersonOnlyLayer),r.traverse(o=>o.layers.set(this._firstPersonOnlyLayer))):i.type==="thirdPersonOnly"?(r.layers.set(this._thirdPersonOnlyLayer),r.traverse(o=>o.layers.set(this._thirdPersonOnlyLayer))):i.type==="auto"&&this._createHeadlessModel(r)})}),this._initializedLayers=!0)}_excludeTriangles(t,n,i,r){let o=0;if(n!=null&&n.length>0)for(let a=0;a<t.length;a+=3){const u=t[a],s=t[a+1],l=t[a+2],d=n[u],c=i[u];if(d[0]>0&&r.includes(c[0])||d[1]>0&&r.includes(c[1])||d[2]>0&&r.includes(c[2])||d[3]>0&&r.includes(c[3]))continue;const p=n[s],h=i[s];if(p[0]>0&&r.includes(h[0])||p[1]>0&&r.includes(h[1])||p[2]>0&&r.includes(h[2])||p[3]>0&&r.includes(h[3]))continue;const _=n[l],g=i[l];_[0]>0&&r.includes(g[0])||_[1]>0&&r.includes(g[1])||_[2]>0&&r.includes(g[2])||_[3]>0&&r.includes(g[3])||(t[o++]=u,t[o++]=s,t[o++]=l)}return o}_createErasedMesh(t,n){const i=new vr(t.geometry.clone(),t.material);i.name=`${t.name}(erase)`,i.frustumCulled=t.frustumCulled,i.layers.set(this._firstPersonOnlyLayer);const r=i.geometry,o=r.getAttribute("skinIndex"),a=o instanceof tn?[]:o.array,u=[];for(let g=0;g<a.length;g+=4)u.push([a[g],a[g+1],a[g+2],a[g+3]]);const s=r.getAttribute("skinWeight"),l=s instanceof tn?[]:s.array,d=[];for(let g=0;g<l.length;g+=4)d.push([l[g],l[g+1],l[g+2],l[g+3]]);const c=r.getIndex();if(!c)throw new Error("The geometry doesn't have an index buffer");const p=Array.from(c.array),h=this._excludeTriangles(p,d,u,n),_=[];for(let g=0;g<h;g++)_[g]=p[g];return r.setIndex(_),t.onBeforeRender&&(i.onBeforeRender=t.onBeforeRender),i.bind(new Tt(t.skeleton.bones,t.skeleton.boneInverses),new J),i}_createHeadlessModelForSkinnedMesh(t,n){const i=[];if(n.skeleton.bones.forEach((o,a)=>{this._isEraseTarget(o)&&i.push(a)}),!i.length){n.layers.enable(this._thirdPersonOnlyLayer),n.layers.enable(this._firstPersonOnlyLayer);return}n.layers.set(this._thirdPersonOnlyLayer);const r=this._createErasedMesh(n,i);t.add(r)}_createHeadlessModel(t){if(t.type==="Group")if(t.layers.set(this._thirdPersonOnlyLayer),this._isEraseTarget(t))t.traverse(n=>n.layers.set(this._thirdPersonOnlyLayer));else{const n=new ve;n.name=`_headless_${t.name}`,n.layers.set(this._firstPersonOnlyLayer),t.parent.add(n),t.children.filter(i=>i.type==="SkinnedMesh").forEach(i=>{const r=i;this._createHeadlessModelForSkinnedMesh(n,r)})}else if(t.type==="SkinnedMesh"){const n=t;this._createHeadlessModelForSkinnedMesh(t.parent,n)}else this._isEraseTarget(t)&&(t.layers.set(this._thirdPersonOnlyLayer),t.traverse(n=>n.layers.set(this._thirdPersonOnlyLayer)))}_isEraseTarget(t){return t===this.humanoid.getRawBoneNode("head")?!0:t.parent?this._isEraseTarget(t.parent):!1}};St.DEFAULT_FIRSTPERSON_ONLY_LAYER=9;St.DEFAULT_THIRDPERSON_ONLY_LAYER=10;var mn=St,lo=new Set(["1.0","1.0-beta"]),uo=class{get name(){return"VRMFirstPersonLoaderPlugin"}constructor(e){this.parser=e}afterRoot(e){return I(this,null,function*(){const t=e.userData.vrmHumanoid;if(t!==null){if(t===void 0)throw new Error("VRMFirstPersonLoaderPlugin: vrmHumanoid is undefined. VRMHumanoidLoaderPlugin have to be used first");e.userData.vrmFirstPerson=yield this._import(e,t)}})}_import(e,t){return I(this,null,function*(){if(t==null)return null;const n=yield this._v1Import(e,t);if(n)return n;const i=yield this._v0Import(e,t);return i||null})}_v1Import(e,t){return I(this,null,function*(){var n,i;const r=this.parser.json;if(!(((n=r.extensionsUsed)==null?void 0:n.indexOf("VRMC_vrm"))!==-1))return null;const a=(i=r.extensions)==null?void 0:i.VRMC_vrm;if(!a)return null;const u=a.specVersion;if(!lo.has(u))return console.warn(`VRMFirstPersonLoaderPlugin: Unknown VRMC_vrm specVersion "${u}"`),null;const s=a.firstPerson,l=[],d=yield dn(e);return Array.from(d.entries()).forEach(([c,p])=>{var h,_;const g=(h=s==null?void 0:s.meshAnnotations)==null?void 0:h.find(m=>m.node===c);l.push({meshes:p,type:(_=g==null?void 0:g.type)!=null?_:"auto"})}),new mn(t,l)})}_v0Import(e,t){return I(this,null,function*(){var n;const i=this.parser.json,r=(n=i.extensions)==null?void 0:n.VRM;if(!r)return null;const o=r.firstPerson;if(!o)return null;const a=[],u=yield dn(e);return Array.from(u.entries()).forEach(([s,l])=>{const d=i.nodes[s],c=o.meshAnnotations?o.meshAnnotations.find(p=>p.mesh===d.mesh):void 0;a.push({meshes:l,type:this._convertV0FlagToV1Type(c==null?void 0:c.firstPersonFlag)})}),new mn(t,a)})}_convertV0FlagToV1Type(e){return e==="FirstPersonOnly"?"firstPersonOnly":e==="ThirdPersonOnly"?"thirdPersonOnly":e==="Both"?"both":"auto"}},_n=new v,gn=new v,co=new S,vn=class extends ve{constructor(e){super(),this.vrmHumanoid=e,this._boneAxesMap=new Map,Object.values(e.humanBones).forEach(t=>{const n=new Mr(1);n.matrixAutoUpdate=!1,n.material.depthTest=!1,n.material.depthWrite=!1,this.add(n),this._boneAxesMap.set(t,n)})}dispose(){Array.from(this._boneAxesMap.values()).forEach(e=>{e.geometry.dispose(),e.material.dispose()})}updateMatrixWorld(e){Array.from(this._boneAxesMap.entries()).forEach(([t,n])=>{t.node.updateWorldMatrix(!0,!1),t.node.matrixWorld.decompose(_n,co,gn);const i=_n.set(.1,.1,.1).divide(gn);n.matrix.copy(t.node.matrixWorld).scale(i)}),super.updateMatrixWorld(e)}},Je=["hips","spine","chest","upperChest","neck","head","leftEye","rightEye","jaw","leftUpperLeg","leftLowerLeg","leftFoot","leftToes","rightUpperLeg","rightLowerLeg","rightFoot","rightToes","leftShoulder","leftUpperArm","leftLowerArm","leftHand","rightShoulder","rightUpperArm","rightLowerArm","rightHand","leftThumbMetacarpal","leftThumbProximal","leftThumbDistal","leftIndexProximal","leftIndexIntermediate","leftIndexDistal","leftMiddleProximal","leftMiddleIntermediate","leftMiddleDistal","leftRingProximal","leftRingIntermediate","leftRingDistal","leftLittleProximal","leftLittleIntermediate","leftLittleDistal","rightThumbMetacarpal","rightThumbProximal","rightThumbDistal","rightIndexProximal","rightIndexIntermediate","rightIndexDistal","rightMiddleProximal","rightMiddleIntermediate","rightMiddleDistal","rightRingProximal","rightRingIntermediate","rightRingDistal","rightLittleProximal","rightLittleIntermediate","rightLittleDistal"],ho={hips:null,spine:"hips",chest:"spine",upperChest:"chest",neck:"upperChest",head:"neck",leftEye:"head",rightEye:"head",jaw:"head",leftUpperLeg:"hips",leftLowerLeg:"leftUpperLeg",leftFoot:"leftLowerLeg",leftToes:"leftFoot",rightUpperLeg:"hips",rightLowerLeg:"rightUpperLeg",rightFoot:"rightLowerLeg",rightToes:"rightFoot",leftShoulder:"upperChest",leftUpperArm:"leftShoulder",leftLowerArm:"leftUpperArm",leftHand:"leftLowerArm",rightShoulder:"upperChest",rightUpperArm:"rightShoulder",rightLowerArm:"rightUpperArm",rightHand:"rightLowerArm",leftThumbMetacarpal:"leftHand",leftThumbProximal:"leftThumbMetacarpal",leftThumbDistal:"leftThumbProximal",leftIndexProximal:"leftHand",leftIndexIntermediate:"leftIndexProximal",leftIndexDistal:"leftIndexIntermediate",leftMiddleProximal:"leftHand",leftMiddleIntermediate:"leftMiddleProximal",leftMiddleDistal:"leftMiddleIntermediate",leftRingProximal:"leftHand",leftRingIntermediate:"leftRingProximal",leftRingDistal:"leftRingIntermediate",leftLittleProximal:"leftHand",leftLittleIntermediate:"leftLittleProximal",leftLittleDistal:"leftLittleIntermediate",rightThumbMetacarpal:"rightHand",rightThumbProximal:"rightThumbMetacarpal",rightThumbDistal:"rightThumbProximal",rightIndexProximal:"rightHand",rightIndexIntermediate:"rightIndexProximal",rightIndexDistal:"rightIndexIntermediate",rightMiddleProximal:"rightHand",rightMiddleIntermediate:"rightMiddleProximal",rightMiddleDistal:"rightMiddleIntermediate",rightRingProximal:"rightHand",rightRingIntermediate:"rightRingProximal",rightRingDistal:"rightRingIntermediate",rightLittleProximal:"rightHand",rightLittleIntermediate:"rightLittleProximal",rightLittleDistal:"rightLittleIntermediate"};function vi(e){return e.invert?e.invert():e.inverse(),e}var oe=new v,se=new S,_t=class{constructor(e){this.humanBones=e,this.restPose=this.getAbsolutePose()}getAbsolutePose(){const e={};return Object.keys(this.humanBones).forEach(t=>{const n=t,i=this.getBoneNode(n);i&&(oe.copy(i.position),se.copy(i.quaternion),e[n]={position:oe.toArray(),rotation:se.toArray()})}),e}getPose(){const e={};return Object.keys(this.humanBones).forEach(t=>{const n=t,i=this.getBoneNode(n);if(!i)return;oe.set(0,0,0),se.identity();const r=this.restPose[n];r!=null&&r.position&&oe.fromArray(r.position).negate(),r!=null&&r.rotation&&vi(se.fromArray(r.rotation)),oe.add(i.position),se.premultiply(i.quaternion),e[n]={position:oe.toArray(),rotation:se.toArray()}}),e}setPose(e){Object.entries(e).forEach(([t,n])=>{const i=t,r=this.getBoneNode(i);if(!r)return;const o=this.restPose[i];o&&(n!=null&&n.position&&(r.position.fromArray(n.position),o.position&&r.position.add(oe.fromArray(o.position))),n!=null&&n.rotation&&(r.quaternion.fromArray(n.rotation),o.rotation&&r.quaternion.multiply(se.fromArray(o.rotation))))})}resetPose(){Object.entries(this.restPose).forEach(([e,t])=>{const n=this.getBoneNode(e);n&&(t!=null&&t.position&&n.position.fromArray(t.position),t!=null&&t.rotation&&n.quaternion.fromArray(t.rotation))})}getBone(e){var t;return(t=this.humanBones[e])!=null?t:void 0}getBoneNode(e){var t,n;return(n=(t=this.humanBones[e])==null?void 0:t.node)!=null?n:null}},Ke=new v,po=new S,fo=new v,Mn=class Mi extends _t{static _setupTransforms(t){const n=new ge;n.name="VRMHumanoidRig";const i={},r={},o={};Je.forEach(u=>{var s;const l=t.getBoneNode(u);if(l){const d=new v,c=new S;l.updateWorldMatrix(!0,!1),l.matrixWorld.decompose(d,c,Ke),i[u]=d,r[u]=l.quaternion.clone();const p=new S;(s=l.parent)==null||s.matrixWorld.decompose(Ke,p,Ke),o[u]=p}});const a={};return Je.forEach(u=>{var s;const l=t.getBoneNode(u);if(l){const d=i[u];let c=u,p;for(;p==null&&(c=ho[c],c!=null);)p=i[c];const h=new ge;h.name="Normalized_"+l.name,(c?(s=a[c])==null?void 0:s.node:n).add(h),h.position.copy(d),p&&h.position.sub(p),a[u]={node:h}}}),{rigBones:a,root:n,parentWorldRotations:o,boneRotations:r}}constructor(t){const{rigBones:n,root:i,parentWorldRotations:r,boneRotations:o}=Mi._setupTransforms(t);super(n),this.original=t,this.root=i,this._parentWorldRotations=r,this._boneRotations=o}update(){Je.forEach(t=>{const n=this.original.getBoneNode(t);if(n!=null){const i=this.getBoneNode(t),r=this._parentWorldRotations[t],o=po.copy(r).invert(),a=this._boneRotations[t];if(n.quaternion.copy(i.quaternion).multiply(r).premultiply(o).multiply(a),t==="hips"){const u=i.getWorldPosition(fo);n.parent.updateWorldMatrix(!0,!1);const s=n.parent.matrixWorld,l=u.applyMatrix4(s.invert());n.position.copy(l)}}})}},xn=class xi{get restPose(){return console.warn("VRMHumanoid: restPose is deprecated. Use either rawRestPose or normalizedRestPose instead."),this.rawRestPose}get rawRestPose(){return this._rawHumanBones.restPose}get normalizedRestPose(){return this._normalizedHumanBones.restPose}get humanBones(){return this._rawHumanBones.humanBones}get rawHumanBones(){return this._rawHumanBones.humanBones}get normalizedHumanBones(){return this._normalizedHumanBones.humanBones}get normalizedHumanBonesRoot(){return this._normalizedHumanBones.root}constructor(t,n){var i;this.autoUpdateHumanBones=(i=n==null?void 0:n.autoUpdateHumanBones)!=null?i:!0,this._rawHumanBones=new _t(t),this._normalizedHumanBones=new Mn(this._rawHumanBones)}copy(t){return this.autoUpdateHumanBones=t.autoUpdateHumanBones,this._rawHumanBones=new _t(t.humanBones),this._normalizedHumanBones=new Mn(this._rawHumanBones),this}clone(){return new xi(this.humanBones,{autoUpdateHumanBones:this.autoUpdateHumanBones}).copy(this)}getAbsolutePose(){return console.warn("VRMHumanoid: getAbsolutePose() is deprecated. Use either getRawAbsolutePose() or getNormalizedAbsolutePose() instead."),this.getRawAbsolutePose()}getRawAbsolutePose(){return this._rawHumanBones.getAbsolutePose()}getNormalizedAbsolutePose(){return this._normalizedHumanBones.getAbsolutePose()}getPose(){return console.warn("VRMHumanoid: getPose() is deprecated. Use either getRawPose() or getNormalizedPose() instead."),this.getRawPose()}getRawPose(){return this._rawHumanBones.getPose()}getNormalizedPose(){return this._normalizedHumanBones.getPose()}setPose(t){return console.warn("VRMHumanoid: setPose() is deprecated. Use either setRawPose() or setNormalizedPose() instead."),this.setRawPose(t)}setRawPose(t){return this._rawHumanBones.setPose(t)}setNormalizedPose(t){return this._normalizedHumanBones.setPose(t)}resetPose(){return console.warn("VRMHumanoid: resetPose() is deprecated. Use either resetRawPose() or resetNormalizedPose() instead."),this.resetRawPose()}resetRawPose(){return this._rawHumanBones.resetPose()}resetNormalizedPose(){return this._normalizedHumanBones.resetPose()}getBone(t){return console.warn("VRMHumanoid: getBone() is deprecated. Use either getRawBone() or getNormalizedBone() instead."),this.getRawBone(t)}getRawBone(t){return this._rawHumanBones.getBone(t)}getNormalizedBone(t){return this._normalizedHumanBones.getBone(t)}getBoneNode(t){return console.warn("VRMHumanoid: getBoneNode() is deprecated. Use either getRawBoneNode() or getNormalizedBoneNode() instead."),this.getRawBoneNode(t)}getRawBoneNode(t){return this._rawHumanBones.getBoneNode(t)}getNormalizedBoneNode(t){return this._normalizedHumanBones.getBoneNode(t)}update(){this.autoUpdateHumanBones&&this._normalizedHumanBones.update()}},mo={Hips:"hips",Spine:"spine",Head:"head",LeftUpperLeg:"leftUpperLeg",LeftLowerLeg:"leftLowerLeg",LeftFoot:"leftFoot",RightUpperLeg:"rightUpperLeg",RightLowerLeg:"rightLowerLeg",RightFoot:"rightFoot",LeftUpperArm:"leftUpperArm",LeftLowerArm:"leftLowerArm",LeftHand:"leftHand",RightUpperArm:"rightUpperArm",RightLowerArm:"rightLowerArm",RightHand:"rightHand"},_o=new Set(["1.0","1.0-beta"]),yn={leftThumbProximal:"leftThumbMetacarpal",leftThumbIntermediate:"leftThumbProximal",rightThumbProximal:"rightThumbMetacarpal",rightThumbIntermediate:"rightThumbProximal"},go=class{get name(){return"VRMHumanoidLoaderPlugin"}constructor(e,t){this.parser=e,this.helperRoot=t==null?void 0:t.helperRoot,this.autoUpdateHumanBones=t==null?void 0:t.autoUpdateHumanBones}afterRoot(e){return I(this,null,function*(){e.userData.vrmHumanoid=yield this._import(e)})}_import(e){return I(this,null,function*(){const t=yield this._v1Import(e);if(t)return t;const n=yield this._v0Import(e);return n||null})}_v1Import(e){return I(this,null,function*(){var t,n;const i=this.parser.json;if(!(((t=i.extensionsUsed)==null?void 0:t.indexOf("VRMC_vrm"))!==-1))return null;const o=(n=i.extensions)==null?void 0:n.VRMC_vrm;if(!o)return null;const a=o.specVersion;if(!_o.has(a))return console.warn(`VRMHumanoidLoaderPlugin: Unknown VRMC_vrm specVersion "${a}"`),null;const u=o.humanoid;if(!u)return null;const s=u.humanBones.leftThumbIntermediate!=null||u.humanBones.rightThumbIntermediate!=null,l={};u.humanBones!=null&&(yield Promise.all(Object.entries(u.humanBones).map(c=>I(this,[c],function*([p,h]){let _=p;const g=h.node;if(s){const f=yn[_];f!=null&&(_=f)}const m=yield this.parser.getDependency("node",g);if(m==null){console.warn(`A glTF node bound to the humanoid bone ${_} (index = ${g}) does not exist`);return}l[_]={node:m}}))));const d=new xn(this._ensureRequiredBonesExist(l),{autoUpdateHumanBones:this.autoUpdateHumanBones});if(e.scene.add(d.normalizedHumanBonesRoot),this.helperRoot){const c=new vn(d);this.helperRoot.add(c),c.renderOrder=this.helperRoot.renderOrder}return d})}_v0Import(e){return I(this,null,function*(){var t;const i=(t=this.parser.json.extensions)==null?void 0:t.VRM;if(!i)return null;const r=i.humanoid;if(!r)return null;const o={};r.humanBones!=null&&(yield Promise.all(r.humanBones.map(u=>I(this,null,function*(){const s=u.bone,l=u.node;if(s==null||l==null)return;if(l<0){console.warn(`A glTF node index for the humanoid bone ${s} is negative (${l}), ignoring this bone.`);return}const d=yield this.parser.getDependency("node",l);if(d==null){console.warn(`A glTF node bound to the humanoid bone ${s} (index = ${l}) does not exist`);return}const c=yn[s],p=c??s;if(o[p]!=null){console.warn(`Multiple bone entries for ${p} detected (index = ${l}), ignoring duplicated entries.`);return}o[p]={node:d}}))));const a=new xn(this._ensureRequiredBonesExist(o),{autoUpdateHumanBones:this.autoUpdateHumanBones});if(e.scene.add(a.normalizedHumanBonesRoot),this.helperRoot){const u=new vn(a);this.helperRoot.add(u),u.renderOrder=this.helperRoot.renderOrder}return a})}_ensureRequiredBonesExist(e){const t=Object.values(mo).filter(n=>e[n]==null);if(t.length>0)throw new Error(`VRMHumanoidLoaderPlugin: These humanoid bones are required but not exist: ${t.join(", ")}`);return e}},wn=class extends ee{constructor(){super(),this._currentTheta=0,this._currentRadius=0,this.theta=0,this.radius=0,this._currentTheta=0,this._currentRadius=0,this._attrPos=new F(new Float32Array(195),3),this.setAttribute("position",this._attrPos),this._attrIndex=new F(new Uint16Array(189),1),this.setIndex(this._attrIndex),this._buildIndex(),this.update()}update(){let e=!1;this._currentTheta!==this.theta&&(this._currentTheta=this.theta,e=!0),this._currentRadius!==this.radius&&(this._currentRadius=this.radius,e=!0),e&&this._buildPosition()}_buildPosition(){this._attrPos.setXYZ(0,0,0,0);for(let e=0;e<64;e++){const t=e/63*this._currentTheta;this._attrPos.setXYZ(e+1,this._currentRadius*Math.sin(t),0,this._currentRadius*Math.cos(t))}this._attrPos.needsUpdate=!0}_buildIndex(){for(let e=0;e<63;e++)this._attrIndex.setXYZ(e*3,0,e+1,e+2);this._attrIndex.needsUpdate=!0}},vo=class extends ee{constructor(){super(),this.radius=0,this._currentRadius=0,this.tail=new v,this._currentTail=new v,this._attrPos=new F(new Float32Array(294),3),this.setAttribute("position",this._attrPos),this._attrIndex=new F(new Uint16Array(194),1),this.setIndex(this._attrIndex),this._buildIndex(),this.update()}update(){let e=!1;this._currentRadius!==this.radius&&(this._currentRadius=this.radius,e=!0),this._currentTail.equals(this.tail)||(this._currentTail.copy(this.tail),e=!0),e&&this._buildPosition()}_buildPosition(){for(let e=0;e<32;e++){const t=e/16*Math.PI;this._attrPos.setXYZ(e,Math.cos(t),Math.sin(t),0),this._attrPos.setXYZ(32+e,0,Math.cos(t),Math.sin(t)),this._attrPos.setXYZ(64+e,Math.sin(t),0,Math.cos(t))}this.scale(this._currentRadius,this._currentRadius,this._currentRadius),this.translate(this._currentTail.x,this._currentTail.y,this._currentTail.z),this._attrPos.setXYZ(96,0,0,0),this._attrPos.setXYZ(97,this._currentTail.x,this._currentTail.y,this._currentTail.z),this._attrPos.needsUpdate=!0}_buildIndex(){for(let e=0;e<32;e++){const t=(e+1)%32;this._attrIndex.setXY(e*2,e,t),this._attrIndex.setXY(64+e*2,32+e,32+t),this._attrIndex.setXY(128+e*2,64+e,64+t)}this._attrIndex.setXY(192,96,97),this._attrIndex.needsUpdate=!0}},Ve=new S,Tn=new S,we=new v,Rn=new v,Sn=Math.sqrt(2)/2,Mo=new S(0,0,-Sn,Sn),xo=new v(0,1,0),yo=class extends ve{constructor(e){super(),this.matrixAutoUpdate=!1,this.vrmLookAt=e;{const t=new wn;t.radius=.5;const n=new Jt({color:65280,transparent:!0,opacity:.5,side:Kt,depthTest:!1,depthWrite:!1});this._meshPitch=new en(t,n),this.add(this._meshPitch)}{const t=new wn;t.radius=.5;const n=new Jt({color:16711680,transparent:!0,opacity:.5,side:Kt,depthTest:!1,depthWrite:!1});this._meshYaw=new en(t,n),this.add(this._meshYaw)}{const t=new vo;t.radius=.1;const n=new qe({color:16777215,depthTest:!1,depthWrite:!1});this._lineTarget=new wt(t,n),this._lineTarget.frustumCulled=!1,this.add(this._lineTarget)}}dispose(){this._meshYaw.geometry.dispose(),this._meshYaw.material.dispose(),this._meshPitch.geometry.dispose(),this._meshPitch.material.dispose(),this._lineTarget.geometry.dispose(),this._lineTarget.material.dispose()}updateMatrixWorld(e){const t=O.DEG2RAD*this.vrmLookAt.yaw;this._meshYaw.geometry.theta=t,this._meshYaw.geometry.update();const n=O.DEG2RAD*this.vrmLookAt.pitch;this._meshPitch.geometry.theta=n,this._meshPitch.geometry.update(),this.vrmLookAt.getLookAtWorldPosition(we),this.vrmLookAt.getLookAtWorldQuaternion(Ve),Ve.multiply(this.vrmLookAt.getFaceFrontQuaternion(Tn)),this._meshYaw.position.copy(we),this._meshYaw.quaternion.copy(Ve),this._meshPitch.position.copy(we),this._meshPitch.quaternion.copy(Ve),this._meshPitch.quaternion.multiply(Tn.setFromAxisAngle(xo,t)),this._meshPitch.quaternion.multiply(Mo);const{target:i,autoUpdate:r}=this.vrmLookAt;i!=null&&r&&(i.getWorldPosition(Rn).sub(we),this._lineTarget.geometry.tail.copy(Rn),this._lineTarget.geometry.update(),this._lineTarget.position.copy(we)),super.updateMatrixWorld(e)}},wo=new v,To=new v;function gt(e,t){return e.matrixWorld.decompose(wo,t,To),t}function Be(e){return[Math.atan2(-e.z,e.x),Math.atan2(e.y,Math.sqrt(e.x*e.x+e.z*e.z))]}function An(e){const t=Math.round(e/2/Math.PI);return e-2*Math.PI*t}var En=new v(0,0,1),Ro=new v,So=new v,Ao=new v,Eo=new S,et=new S,Pn=new S,Po=new S,tt=new ie,yi=class wi{constructor(t,n){this.offsetFromHeadBone=new v,this.autoUpdate=!0,this.faceFront=new v(0,0,1),this.humanoid=t,this.applier=n,this._yaw=0,this._pitch=0,this._needsUpdate=!0,this._restHeadWorldQuaternion=this.getLookAtWorldQuaternion(new S)}get yaw(){return this._yaw}set yaw(t){this._yaw=t,this._needsUpdate=!0}get pitch(){return this._pitch}set pitch(t){this._pitch=t,this._needsUpdate=!0}get euler(){return console.warn("VRMLookAt: euler is deprecated. use getEuler() instead."),this.getEuler(new ie)}getEuler(t){return t.set(O.DEG2RAD*this._pitch,O.DEG2RAD*this._yaw,0,"YXZ")}copy(t){if(this.humanoid!==t.humanoid)throw new Error("VRMLookAt: humanoid must be same in order to copy");return this.offsetFromHeadBone.copy(t.offsetFromHeadBone),this.applier=t.applier,this.autoUpdate=t.autoUpdate,this.target=t.target,this.faceFront.copy(t.faceFront),this}clone(){return new wi(this.humanoid,this.applier).copy(this)}reset(){this._yaw=0,this._pitch=0,this._needsUpdate=!0}getLookAtWorldPosition(t){const n=this.humanoid.getRawBoneNode("head");return t.copy(this.offsetFromHeadBone).applyMatrix4(n.matrixWorld)}getLookAtWorldQuaternion(t){const n=this.humanoid.getRawBoneNode("head");return gt(n,t)}getFaceFrontQuaternion(t){if(this.faceFront.distanceToSquared(En)<.01)return t.copy(this._restHeadWorldQuaternion).invert();const[n,i]=Be(this.faceFront);return tt.set(0,.5*Math.PI+n,i,"YZX"),t.setFromEuler(tt).premultiply(Po.copy(this._restHeadWorldQuaternion).invert())}getLookAtWorldDirection(t){return this.getLookAtWorldQuaternion(et),this.getFaceFrontQuaternion(Pn),t.copy(En).applyQuaternion(et).applyQuaternion(Pn).applyEuler(this.getEuler(tt))}lookAt(t){const n=Eo.copy(this._restHeadWorldQuaternion).multiply(vi(this.getLookAtWorldQuaternion(et))),i=this.getLookAtWorldPosition(So),r=Ao.copy(t).sub(i).applyQuaternion(n).normalize(),[o,a]=Be(this.faceFront),[u,s]=Be(r),l=An(u-o),d=An(a-s);this._yaw=O.RAD2DEG*l,this._pitch=O.RAD2DEG*d,this._needsUpdate=!0}update(t){this.target!=null&&this.autoUpdate&&this.lookAt(this.target.getWorldPosition(Ro)),this._needsUpdate&&(this._needsUpdate=!1,this.applier.applyYawPitch(this._yaw,this._pitch))}};yi.EULER_ORDER="YXZ";var Lo=yi,bo=new v(0,0,1),q=new S,de=new S,j=new ie(0,0,0,"YXZ"),Fe=class{constructor(e,t,n,i,r){this.humanoid=e,this.rangeMapHorizontalInner=t,this.rangeMapHorizontalOuter=n,this.rangeMapVerticalDown=i,this.rangeMapVerticalUp=r,this.faceFront=new v(0,0,1),this._restQuatLeftEye=new S,this._restQuatRightEye=new S,this._restLeftEyeParentWorldQuat=new S,this._restRightEyeParentWorldQuat=new S;const o=this.humanoid.getRawBoneNode("leftEye"),a=this.humanoid.getRawBoneNode("rightEye");o&&(this._restQuatLeftEye.copy(o.quaternion),gt(o.parent,this._restLeftEyeParentWorldQuat)),a&&(this._restQuatRightEye.copy(a.quaternion),gt(a.parent,this._restRightEyeParentWorldQuat))}applyYawPitch(e,t){const n=this.humanoid.getRawBoneNode("leftEye"),i=this.humanoid.getRawBoneNode("rightEye"),r=this.humanoid.getNormalizedBoneNode("leftEye"),o=this.humanoid.getNormalizedBoneNode("rightEye");n&&(t<0?j.x=-O.DEG2RAD*this.rangeMapVerticalDown.map(-t):j.x=O.DEG2RAD*this.rangeMapVerticalUp.map(t),e<0?j.y=-O.DEG2RAD*this.rangeMapHorizontalInner.map(-e):j.y=O.DEG2RAD*this.rangeMapHorizontalOuter.map(e),q.setFromEuler(j),this._getWorldFaceFrontQuat(de),r.quaternion.copy(de).multiply(q).multiply(de.invert()),q.copy(this._restLeftEyeParentWorldQuat),n.quaternion.copy(r.quaternion).multiply(q).premultiply(q.invert()).multiply(this._restQuatLeftEye)),i&&(t<0?j.x=-O.DEG2RAD*this.rangeMapVerticalDown.map(-t):j.x=O.DEG2RAD*this.rangeMapVerticalUp.map(t),e<0?j.y=-O.DEG2RAD*this.rangeMapHorizontalOuter.map(-e):j.y=O.DEG2RAD*this.rangeMapHorizontalInner.map(e),q.setFromEuler(j),this._getWorldFaceFrontQuat(de),o.quaternion.copy(de).multiply(q).multiply(de.invert()),q.copy(this._restRightEyeParentWorldQuat),i.quaternion.copy(o.quaternion).multiply(q).premultiply(q.invert()).multiply(this._restQuatRightEye))}lookAt(e){console.warn("VRMLookAtBoneApplier: lookAt() is deprecated. use apply() instead.");const t=O.RAD2DEG*e.y,n=O.RAD2DEG*e.x;this.applyYawPitch(t,n)}_getWorldFaceFrontQuat(e){if(this.faceFront.distanceToSquared(bo)<.01)return e.identity();const[t,n]=Be(this.faceFront);return j.set(0,.5*Math.PI+t,n,"YZX"),e.setFromEuler(j)}};Fe.type="bone";var vt=class{constructor(e,t,n,i,r){this.expressions=e,this.rangeMapHorizontalInner=t,this.rangeMapHorizontalOuter=n,this.rangeMapVerticalDown=i,this.rangeMapVerticalUp=r}applyYawPitch(e,t){t<0?(this.expressions.setValue("lookDown",0),this.expressions.setValue("lookUp",this.rangeMapVerticalUp.map(-t))):(this.expressions.setValue("lookUp",0),this.expressions.setValue("lookDown",this.rangeMapVerticalDown.map(t))),e<0?(this.expressions.setValue("lookLeft",0),this.expressions.setValue("lookRight",this.rangeMapHorizontalOuter.map(-e))):(this.expressions.setValue("lookRight",0),this.expressions.setValue("lookLeft",this.rangeMapHorizontalOuter.map(e)))}lookAt(e){console.warn("VRMLookAtBoneApplier: lookAt() is deprecated. use apply() instead.");const t=O.RAD2DEG*e.y,n=O.RAD2DEG*e.x;this.applyYawPitch(t,n)}};vt.type="expression";var Ln=class{constructor(e,t){this.inputMaxValue=e,this.outputScale=t}map(e){return this.outputScale*di(e/this.inputMaxValue)}},Io=new Set(["1.0","1.0-beta"]),ke=.01,Co=class{get name(){return"VRMLookAtLoaderPlugin"}constructor(e,t){this.parser=e,this.helperRoot=t==null?void 0:t.helperRoot}afterRoot(e){return I(this,null,function*(){const t=e.userData.vrmHumanoid;if(t===null)return;if(t===void 0)throw new Error("VRMLookAtLoaderPlugin: vrmHumanoid is undefined. VRMHumanoidLoaderPlugin have to be used first");const n=e.userData.vrmExpressionManager;if(n!==null){if(n===void 0)throw new Error("VRMLookAtLoaderPlugin: vrmExpressionManager is undefined. VRMExpressionLoaderPlugin have to be used first");e.userData.vrmLookAt=yield this._import(e,t,n)}})}_import(e,t,n){return I(this,null,function*(){if(t==null||n==null)return null;const i=yield this._v1Import(e,t,n);if(i)return i;const r=yield this._v0Import(e,t,n);return r||null})}_v1Import(e,t,n){return I(this,null,function*(){var i,r,o;const a=this.parser.json;if(!(((i=a.extensionsUsed)==null?void 0:i.indexOf("VRMC_vrm"))!==-1))return null;const s=(r=a.extensions)==null?void 0:r.VRMC_vrm;if(!s)return null;const l=s.specVersion;if(!Io.has(l))return console.warn(`VRMLookAtLoaderPlugin: Unknown VRMC_vrm specVersion "${l}"`),null;const d=s.lookAt;if(!d)return null;const c=d.type==="expression"?1:10,p=this._v1ImportRangeMap(d.rangeMapHorizontalInner,c),h=this._v1ImportRangeMap(d.rangeMapHorizontalOuter,c),_=this._v1ImportRangeMap(d.rangeMapVerticalDown,c),g=this._v1ImportRangeMap(d.rangeMapVerticalUp,c);let m;d.type==="expression"?m=new vt(n,p,h,_,g):m=new Fe(t,p,h,_,g);const f=this._importLookAt(t,m);return f.offsetFromHeadBone.fromArray((o=d.offsetFromHeadBone)!=null?o:[0,.06,0]),f})}_v1ImportRangeMap(e,t){var n,i;let r=(n=e==null?void 0:e.inputMaxValue)!=null?n:90;const o=(i=e==null?void 0:e.outputScale)!=null?i:t;return r<ke&&(console.warn("VRMLookAtLoaderPlugin: inputMaxValue of a range map is too small. Consider reviewing the range map!"),r=ke),new Ln(r,o)}_v0Import(e,t,n){return I(this,null,function*(){var i,r,o,a;const s=(i=this.parser.json.extensions)==null?void 0:i.VRM;if(!s)return null;const l=s.firstPerson;if(!l)return null;const d=l.lookAtTypeName==="BlendShape"?1:10,c=this._v0ImportDegreeMap(l.lookAtHorizontalInner,d),p=this._v0ImportDegreeMap(l.lookAtHorizontalOuter,d),h=this._v0ImportDegreeMap(l.lookAtVerticalDown,d),_=this._v0ImportDegreeMap(l.lookAtVerticalUp,d);let g;l.lookAtTypeName==="BlendShape"?g=new vt(n,c,p,h,_):g=new Fe(t,c,p,h,_);const m=this._importLookAt(t,g);return l.firstPersonBoneOffset?m.offsetFromHeadBone.set((r=l.firstPersonBoneOffset.x)!=null?r:0,(o=l.firstPersonBoneOffset.y)!=null?o:.06,-((a=l.firstPersonBoneOffset.z)!=null?a:0)):m.offsetFromHeadBone.set(0,.06,0),m.faceFront.set(0,0,-1),g instanceof Fe&&g.faceFront.set(0,0,-1),m})}_v0ImportDegreeMap(e,t){var n,i;const r=e==null?void 0:e.curve;JSON.stringify(r)!=="[0,0,0,1,1,1,1,0]"&&console.warn("Curves of LookAtDegreeMap defined in VRM 0.0 are not supported");let o=(n=e==null?void 0:e.xRange)!=null?n:90;const a=(i=e==null?void 0:e.yRange)!=null?i:t;return o<ke&&(console.warn("VRMLookAtLoaderPlugin: xRange of a degree map is too small. Consider reviewing the degree map!"),o=ke),new Ln(o,a)}_importLookAt(e,t){const n=new Lo(e,t);if(this.helperRoot){const i=new yo(n);this.helperRoot.add(i),i.renderOrder=this.helperRoot.renderOrder}return n}};function No(e,t){return typeof e!="string"||e===""?"":(/^https?:\/\//i.test(t)&&/^\//.test(e)&&(t=t.replace(/(^https?:\/\/[^/]+).*/i,"$1")),/^(https?:)?\/\//i.test(e)||/^data:.*,.*$/i.test(e)||/^blob:.*$/i.test(e)?e:t+e)}var Oo=new Set(["1.0","1.0-beta"]),Uo=class{get name(){return"VRMMetaLoaderPlugin"}constructor(e,t){var n,i,r;this.parser=e,this.needThumbnailImage=(n=t==null?void 0:t.needThumbnailImage)!=null?n:!1,this.acceptLicenseUrls=(i=t==null?void 0:t.acceptLicenseUrls)!=null?i:["https://vrm.dev/licenses/1.0/"],this.acceptV0Meta=(r=t==null?void 0:t.acceptV0Meta)!=null?r:!0}afterRoot(e){return I(this,null,function*(){e.userData.vrmMeta=yield this._import(e)})}_import(e){return I(this,null,function*(){const t=yield this._v1Import(e);if(t!=null)return t;const n=yield this._v0Import(e);return n??null})}_v1Import(e){return I(this,null,function*(){var t,n,i;const r=this.parser.json;if(!(((t=r.extensionsUsed)==null?void 0:t.indexOf("VRMC_vrm"))!==-1))return null;const a=(n=r.extensions)==null?void 0:n.VRMC_vrm;if(a==null)return null;const u=a.specVersion;if(!Oo.has(u))return console.warn(`VRMMetaLoaderPlugin: Unknown VRMC_vrm specVersion "${u}"`),null;const s=a.meta;if(!s)return null;const l=s.licenseUrl;if(!new Set(this.acceptLicenseUrls).has(l))throw new Error(`VRMMetaLoaderPlugin: The license url "${l}" is not accepted`);let c;return this.needThumbnailImage&&s.thumbnailImage!=null&&(c=(i=yield this._extractGLTFImage(s.thumbnailImage))!=null?i:void 0),{metaVersion:"1",name:s.name,version:s.version,authors:s.authors,copyrightInformation:s.copyrightInformation,contactInformation:s.contactInformation,references:s.references,thirdPartyLicenses:s.thirdPartyLicenses,thumbnailImage:c,licenseUrl:s.licenseUrl,avatarPermission:s.avatarPermission,allowExcessivelyViolentUsage:s.allowExcessivelyViolentUsage,allowExcessivelySexualUsage:s.allowExcessivelySexualUsage,commercialUsage:s.commercialUsage,allowPoliticalOrReligiousUsage:s.allowPoliticalOrReligiousUsage,allowAntisocialOrHateUsage:s.allowAntisocialOrHateUsage,creditNotation:s.creditNotation,allowRedistribution:s.allowRedistribution,modification:s.modification,otherLicenseUrl:s.otherLicenseUrl}})}_v0Import(e){return I(this,null,function*(){var t;const i=(t=this.parser.json.extensions)==null?void 0:t.VRM;if(!i)return null;const r=i.meta;if(!r)return null;if(!this.acceptV0Meta)throw new Error("VRMMetaLoaderPlugin: Attempted to load VRM0.0 meta but acceptV0Meta is false");let o;return this.needThumbnailImage&&r.texture!=null&&r.texture!==-1&&(o=yield this.parser.getDependency("texture",r.texture)),{metaVersion:"0",allowedUserName:r.allowedUserName,author:r.author,commercialUssageName:r.commercialUssageName,contactInformation:r.contactInformation,licenseName:r.licenseName,otherLicenseUrl:r.otherLicenseUrl,otherPermissionUrl:r.otherPermissionUrl,reference:r.reference,sexualUssageName:r.sexualUssageName,texture:o??void 0,title:r.title,version:r.version,violentUssageName:r.violentUssageName}})}_extractGLTFImage(e){return I(this,null,function*(){var t;const i=(t=this.parser.json.images)==null?void 0:t[e];if(i==null)return console.warn(`VRMMetaLoaderPlugin: Attempt to use images[${e}] of glTF as a thumbnail but the image doesn't exist`),null;let r=i.uri;if(i.bufferView!=null){const a=yield this.parser.getDependency("bufferView",i.bufferView),u=new Blob([a],{type:i.mimeType});r=URL.createObjectURL(u)}return r==null?(console.warn(`VRMMetaLoaderPlugin: Attempt to use images[${e}] of glTF as a thumbnail but the image couldn't load properly`),null):yield new hr().loadAsync(No(r,this.parser.options.path)).catch(a=>(console.error(a),console.warn("VRMMetaLoaderPlugin: Failed to load a thumbnail image"),null))})}},Vo=class{constructor(e){this.scene=e.scene,this.meta=e.meta,this.humanoid=e.humanoid,this.expressionManager=e.expressionManager,this.firstPerson=e.firstPerson,this.lookAt=e.lookAt}update(e){this.humanoid.update(),this.lookAt&&this.lookAt.update(e),this.expressionManager&&this.expressionManager.update()}},ko=class extends Vo{constructor(e){super(e),this.materials=e.materials,this.springBoneManager=e.springBoneManager,this.nodeConstraintManager=e.nodeConstraintManager}update(e){super.update(e),this.nodeConstraintManager&&this.nodeConstraintManager.update(),this.springBoneManager&&this.springBoneManager.update(e),this.materials&&this.materials.forEach(t=>{t.update&&t.update(e)})}},Do=Object.defineProperty,bn=Object.getOwnPropertySymbols,Bo=Object.prototype.hasOwnProperty,Fo=Object.prototype.propertyIsEnumerable,In=(e,t,n)=>t in e?Do(e,t,{enumerable:!0,configurable:!0,writable:!0,value:n}):e[t]=n,Cn=(e,t)=>{for(var n in t||(t={}))Bo.call(t,n)&&In(e,n,t[n]);if(bn)for(var n of bn(t))Fo.call(t,n)&&In(e,n,t[n]);return e},le=(e,t,n)=>new Promise((i,r)=>{var o=s=>{try{u(n.next(s))}catch(l){r(l)}},a=s=>{try{u(n.throw(s))}catch(l){r(l)}},u=s=>s.done?i(s.value):Promise.resolve(s.value).then(o,a);u((n=n.apply(e,t)).next())}),Ho={"":3e3,srgb:3001};function Wo(e,t){parseInt(We,10)>=152?e.colorSpace=t:e.encoding=Ho[t]}var zo=class{get pending(){return Promise.all(this._pendings)}constructor(e,t){this._parser=e,this._materialParams=t,this._pendings=[]}assignPrimitive(e,t){t!=null&&(this._materialParams[e]=t)}assignColor(e,t,n){if(t!=null){const i=new H().fromArray(t);n&&i.convertSRGBToLinear(),this._materialParams[e]=i}}assignTexture(e,t,n){return le(this,null,function*(){const i=le(this,null,function*(){if(t!=null){const r=yield this._parser.assignTexture(this._materialParams,e,t);if(r==null){console.warn("GLTFMToonMaterialParamsAssignHelper: Failed to load texture. The rendering result may be wrong");return}n&&Wo(r,"srgb")}});return this._pendings.push(i),i})}assignTextureByIndex(e,t,n){return le(this,null,function*(){return this.assignTexture(e,t!=null?{index:t}:void 0,n)})}},jo=`// #define PHONG

varying vec3 vViewPosition;

#ifndef FLAT_SHADED
  varying vec3 vNormal;
#endif

#include <common>

// #include <uv_pars_vertex>
#ifdef MTOON_USE_UV
  varying vec2 vUv;

  // COMPAT: pre-r151 uses a common uvTransform
  #if THREE_VRM_THREE_REVISION < 151
    uniform mat3 uvTransform;
  #endif
#endif

// #include <uv2_pars_vertex>
// COMAPT: pre-r151 uses uv2 for lightMap and aoMap
#if THREE_VRM_THREE_REVISION < 151
  #if defined( USE_LIGHTMAP ) || defined( USE_AOMAP )
    attribute vec2 uv2;
    varying vec2 vUv2;
    uniform mat3 uv2Transform;
  #endif
#endif

// #include <displacementmap_pars_vertex>
// #include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>

#ifdef USE_OUTLINEWIDTHMULTIPLYTEXTURE
  uniform sampler2D outlineWidthMultiplyTexture;
  uniform mat3 outlineWidthMultiplyTextureUvTransform;
#endif

uniform float outlineWidthFactor;

void main() {

  // #include <uv_vertex>
  #ifdef MTOON_USE_UV
    // COMPAT: pre-r151 uses a common uvTransform
    #if THREE_VRM_THREE_REVISION >= 151
      vUv = uv;
    #else
      vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
    #endif
  #endif

  // #include <uv2_vertex>
  // COMAPT: pre-r151 uses uv2 for lightMap and aoMap
  #if THREE_VRM_THREE_REVISION < 151
    #if defined( USE_LIGHTMAP ) || defined( USE_AOMAP )
      vUv2 = ( uv2Transform * vec3( uv2, 1 ) ).xy;
    #endif
  #endif

  #include <color_vertex>

  #include <beginnormal_vertex>
  #include <morphnormal_vertex>
  #include <skinbase_vertex>
  #include <skinnormal_vertex>

  // we need this to compute the outline properly
  objectNormal = normalize( objectNormal );

  #include <defaultnormal_vertex>

  #ifndef FLAT_SHADED // Normal computed with derivatives when FLAT_SHADED
    vNormal = normalize( transformedNormal );
  #endif

  #include <begin_vertex>

  #include <morphtarget_vertex>
  #include <skinning_vertex>
  // #include <displacementmap_vertex>
  #include <project_vertex>
  #include <logdepthbuf_vertex>
  #include <clipping_planes_vertex>

  vViewPosition = - mvPosition.xyz;

  #ifdef OUTLINE
    float worldNormalLength = length( transformedNormal );
    vec3 outlineOffset = outlineWidthFactor * worldNormalLength * objectNormal;

    #ifdef USE_OUTLINEWIDTHMULTIPLYTEXTURE
      vec2 outlineWidthMultiplyTextureUv = ( outlineWidthMultiplyTextureUvTransform * vec3( vUv, 1 ) ).xy;
      float outlineTex = texture2D( outlineWidthMultiplyTexture, outlineWidthMultiplyTextureUv ).g;
      outlineOffset *= outlineTex;
    #endif

    #ifdef OUTLINE_WIDTH_SCREEN
      outlineOffset *= vViewPosition.z / projectionMatrix[ 1 ].y;
    #endif

    gl_Position = projectionMatrix * modelViewMatrix * vec4( outlineOffset + transformed, 1.0 );

    gl_Position.z += 1E-6 * gl_Position.w; // anti-artifact magic
  #endif

  #include <worldpos_vertex>
  // #include <envmap_vertex>
  #include <shadowmap_vertex>
  #include <fog_vertex>

}`,Go=`// #define PHONG

uniform vec3 litFactor;

uniform float opacity;

uniform vec3 shadeColorFactor;
#ifdef USE_SHADEMULTIPLYTEXTURE
  uniform sampler2D shadeMultiplyTexture;
  uniform mat3 shadeMultiplyTextureUvTransform;
#endif

uniform float shadingShiftFactor;
uniform float shadingToonyFactor;

#ifdef USE_SHADINGSHIFTTEXTURE
  uniform sampler2D shadingShiftTexture;
  uniform mat3 shadingShiftTextureUvTransform;
  uniform float shadingShiftTextureScale;
#endif

uniform float giEqualizationFactor;

uniform vec3 parametricRimColorFactor;
#ifdef USE_RIMMULTIPLYTEXTURE
  uniform sampler2D rimMultiplyTexture;
  uniform mat3 rimMultiplyTextureUvTransform;
#endif
uniform float rimLightingMixFactor;
uniform float parametricRimFresnelPowerFactor;
uniform float parametricRimLiftFactor;

#ifdef USE_MATCAPTEXTURE
  uniform vec3 matcapFactor;
  uniform sampler2D matcapTexture;
  uniform mat3 matcapTextureUvTransform;
#endif

uniform vec3 emissive;
uniform float emissiveIntensity;

uniform vec3 outlineColorFactor;
uniform float outlineLightingMixFactor;

#ifdef USE_UVANIMATIONMASKTEXTURE
  uniform sampler2D uvAnimationMaskTexture;
  uniform mat3 uvAnimationMaskTextureUvTransform;
#endif

uniform float uvAnimationScrollXOffset;
uniform float uvAnimationScrollYOffset;
uniform float uvAnimationRotationPhase;

#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>

// #include <uv_pars_fragment>
#if ( defined( MTOON_USE_UV ) && !defined( MTOON_UVS_VERTEX_ONLY ) )
  varying vec2 vUv;
#endif

// #include <uv2_pars_fragment>
// COMAPT: pre-r151 uses uv2 for lightMap and aoMap
#if THREE_VRM_THREE_REVISION < 151
  #if defined( USE_LIGHTMAP ) || defined( USE_AOMAP )
    varying vec2 vUv2;
  #endif
#endif

#include <map_pars_fragment>

#ifdef USE_MAP
  uniform mat3 mapUvTransform;
#endif

// #include <alphamap_pars_fragment>

#include <alphatest_pars_fragment>

#include <aomap_pars_fragment>
// #include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>

#ifdef USE_EMISSIVEMAP
  uniform mat3 emissiveMapUvTransform;
#endif

// #include <envmap_common_pars_fragment>
// #include <envmap_pars_fragment>
// #include <cube_uv_reflection_fragment>
#include <fog_pars_fragment>

// #include <bsdfs>
// COMPAT: pre-r151 doesn't have BRDF_Lambert in <common>
#if THREE_VRM_THREE_REVISION < 151
  vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
    return RECIPROCAL_PI * diffuseColor;
  }
#endif

#include <lights_pars_begin>

#include <normal_pars_fragment>

// #include <lights_phong_pars_fragment>
varying vec3 vViewPosition;

struct MToonMaterial {
  vec3 diffuseColor;
  vec3 shadeColor;
  float shadingShift;
};

float linearstep( float a, float b, float t ) {
  return clamp( ( t - a ) / ( b - a ), 0.0, 1.0 );
}

/**
 * Convert NdotL into toon shading factor using shadingShift and shadingToony
 */
float getShading(
  const in float dotNL,
  const in float shadow,
  const in float shadingShift
) {
  float shading = dotNL;
  shading = shading + shadingShift;
  shading = linearstep( -1.0 + shadingToonyFactor, 1.0 - shadingToonyFactor, shading );
  shading *= shadow;
  return shading;
}

/**
 * Mix diffuseColor and shadeColor using shading factor and light color
 */
vec3 getDiffuse(
  const in MToonMaterial material,
  const in float shading,
  in vec3 lightColor
) {
  #ifdef DEBUG_LITSHADERATE
    return vec3( BRDF_Lambert( shading * lightColor ) );
  #endif

  vec3 col = lightColor * BRDF_Lambert( mix( material.shadeColor, material.diffuseColor, shading ) );

  // The "comment out if you want to PBR absolutely" line
  #ifdef V0_COMPAT_SHADE
    col = min( col, material.diffuseColor );
  #endif

  return col;
}

// COMPAT: pre-r156 uses a struct GeometricContext
#if THREE_VRM_THREE_REVISION >= 157
  void RE_Direct_MToon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in MToonMaterial material, const in float shadow, inout ReflectedLight reflectedLight ) {
    float dotNL = clamp( dot( geometryNormal, directLight.direction ), -1.0, 1.0 );
    vec3 irradiance = directLight.color;

    // directSpecular will be used for rim lighting, not an actual specular
    reflectedLight.directSpecular += irradiance;

    irradiance *= dotNL;

    float shading = getShading( dotNL, shadow, material.shadingShift );

    // toon shaded diffuse
    reflectedLight.directDiffuse += getDiffuse( material, shading, directLight.color );
  }

  void RE_IndirectDiffuse_MToon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in MToonMaterial material, inout ReflectedLight reflectedLight ) {
    // indirect diffuse will use diffuseColor, no shadeColor involved
    reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );

    // directSpecular will be used for rim lighting, not an actual specular
    reflectedLight.directSpecular += irradiance;
  }
#else
  void RE_Direct_MToon( const in IncidentLight directLight, const in GeometricContext geometry, const in MToonMaterial material, const in float shadow, inout ReflectedLight reflectedLight ) {
    float dotNL = clamp( dot( geometry.normal, directLight.direction ), -1.0, 1.0 );
    vec3 irradiance = directLight.color;

    // directSpecular will be used for rim lighting, not an actual specular
    reflectedLight.directSpecular += irradiance;

    irradiance *= dotNL;

    float shading = getShading( dotNL, shadow, material.shadingShift );

    // toon shaded diffuse
    reflectedLight.directDiffuse += getDiffuse( material, shading, directLight.color );
  }

  void RE_IndirectDiffuse_MToon( const in vec3 irradiance, const in GeometricContext geometry, const in MToonMaterial material, inout ReflectedLight reflectedLight ) {
    // indirect diffuse will use diffuseColor, no shadeColor involved
    reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );

    // directSpecular will be used for rim lighting, not an actual specular
    reflectedLight.directSpecular += irradiance;
  }
#endif

#define RE_Direct RE_Direct_MToon
#define RE_IndirectDiffuse RE_IndirectDiffuse_MToon
#define Material_LightProbeLOD( material ) (0)

#include <shadowmap_pars_fragment>
// #include <bumpmap_pars_fragment>

// #include <normalmap_pars_fragment>
#ifdef USE_NORMALMAP

  uniform sampler2D normalMap;
  uniform mat3 normalMapUvTransform;
  uniform vec2 normalScale;

#endif

// COMPAT: pre-r151
// USE_NORMALMAP_OBJECTSPACE used to be OBJECTSPACE_NORMALMAP in pre-r151
#if defined( USE_NORMALMAP_OBJECTSPACE ) || defined( OBJECTSPACE_NORMALMAP )

  uniform mat3 normalMatrix;

#endif

// COMPAT: pre-r151
// USE_NORMALMAP_TANGENTSPACE used to be TANGENTSPACE_NORMALMAP in pre-r151
#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( TANGENTSPACE_NORMALMAP ) )

  // Per-Pixel Tangent Space Normal Mapping
  // http://hacksoflife.blogspot.ch/2009/11/per-pixel-tangent-space-normal-mapping.html

  // three-vrm specific change: it requires \`uv\` as an input in order to support uv scrolls

  // Temporary compat against shader change @ Three.js r126, r151
  #if THREE_VRM_THREE_REVISION >= 151

    mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {

      vec3 q0 = dFdx( eye_pos.xyz );
      vec3 q1 = dFdy( eye_pos.xyz );
      vec2 st0 = dFdx( uv.st );
      vec2 st1 = dFdy( uv.st );

      vec3 N = surf_norm;

      vec3 q1perp = cross( q1, N );
      vec3 q0perp = cross( N, q0 );

      vec3 T = q1perp * st0.x + q0perp * st1.x;
      vec3 B = q1perp * st0.y + q0perp * st1.y;

      float det = max( dot( T, T ), dot( B, B ) );
      float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );

      return mat3( T * scale, B * scale, N );

    }

  #else

    vec3 perturbNormal2Arb( vec2 uv, vec3 eye_pos, vec3 surf_norm, vec3 mapN, float faceDirection ) {

      vec3 q0 = vec3( dFdx( eye_pos.x ), dFdx( eye_pos.y ), dFdx( eye_pos.z ) );
      vec3 q1 = vec3( dFdy( eye_pos.x ), dFdy( eye_pos.y ), dFdy( eye_pos.z ) );
      vec2 st0 = dFdx( uv.st );
      vec2 st1 = dFdy( uv.st );

      vec3 N = normalize( surf_norm );

      vec3 q1perp = cross( q1, N );
      vec3 q0perp = cross( N, q0 );

      vec3 T = q1perp * st0.x + q0perp * st1.x;
      vec3 B = q1perp * st0.y + q0perp * st1.y;

      // three-vrm specific change: Workaround for the issue that happens when delta of uv = 0.0
      // TODO: Is this still required? Or shall I make a PR about it?
      if ( length( T ) == 0.0 || length( B ) == 0.0 ) {
        return surf_norm;
      }

      float det = max( dot( T, T ), dot( B, B ) );
      float scale = ( det == 0.0 ) ? 0.0 : faceDirection * inversesqrt( det );

      return normalize( T * ( mapN.x * scale ) + B * ( mapN.y * scale ) + N * mapN.z );

    }

  #endif

#endif

// #include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>

// == post correction ==========================================================
void postCorrection() {
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
  #include <fog_fragment>
  #include <premultiplied_alpha_fragment>
  #include <dithering_fragment>
}

// == main procedure ===========================================================
void main() {
  #include <clipping_planes_fragment>

  vec2 uv = vec2(0.5, 0.5);

  #if ( defined( MTOON_USE_UV ) && !defined( MTOON_UVS_VERTEX_ONLY ) )
    uv = vUv;

    float uvAnimMask = 1.0;
    #ifdef USE_UVANIMATIONMASKTEXTURE
      vec2 uvAnimationMaskTextureUv = ( uvAnimationMaskTextureUvTransform * vec3( uv, 1 ) ).xy;
      uvAnimMask = texture2D( uvAnimationMaskTexture, uvAnimationMaskTextureUv ).b;
    #endif

    float uvRotCos = cos( uvAnimationRotationPhase * uvAnimMask );
    float uvRotSin = sin( uvAnimationRotationPhase * uvAnimMask );
    uv = mat2( uvRotCos, -uvRotSin, uvRotSin, uvRotCos ) * ( uv - 0.5 ) + 0.5;
    uv = uv + vec2( uvAnimationScrollXOffset, uvAnimationScrollYOffset ) * uvAnimMask;
  #endif

  #ifdef DEBUG_UV
    gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
    #if ( defined( MTOON_USE_UV ) && !defined( MTOON_UVS_VERTEX_ONLY ) )
      gl_FragColor = vec4( uv, 0.0, 1.0 );
    #endif
    return;
  #endif

  vec4 diffuseColor = vec4( litFactor, opacity );
  ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
  vec3 totalEmissiveRadiance = emissive * emissiveIntensity;

  #include <logdepthbuf_fragment>

  // #include <map_fragment>
  #ifdef USE_MAP
    vec2 mapUv = ( mapUvTransform * vec3( uv, 1 ) ).xy;
    vec4 sampledDiffuseColor = texture2D( map, mapUv );
    #ifdef DECODE_VIDEO_TEXTURE
      sampledDiffuseColor = vec4( mix( pow( sampledDiffuseColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), sampledDiffuseColor.rgb * 0.0773993808, vec3( lessThanEqual( sampledDiffuseColor.rgb, vec3( 0.04045 ) ) ) ), sampledDiffuseColor.w );
    #endif
    diffuseColor *= sampledDiffuseColor;
  #endif

  // #include <color_fragment>
  #if ( defined( USE_COLOR ) && !defined( IGNORE_VERTEX_COLOR ) )
    diffuseColor.rgb *= vColor;
  #endif

  // #include <alphamap_fragment>

  #include <alphatest_fragment>

  // #include <specularmap_fragment>

  // #include <normal_fragment_begin>
  float faceDirection = gl_FrontFacing ? 1.0 : -1.0;

  #ifdef FLAT_SHADED

    vec3 fdx = dFdx( vViewPosition );
    vec3 fdy = dFdy( vViewPosition );
    vec3 normal = normalize( cross( fdx, fdy ) );

  #else

    vec3 normal = normalize( vNormal );

    #ifdef DOUBLE_SIDED

      normal *= faceDirection;

    #endif

  #endif

  #ifdef USE_NORMALMAP

    vec2 normalMapUv = ( normalMapUvTransform * vec3( uv, 1 ) ).xy;

  #endif

  #ifdef USE_NORMALMAP_TANGENTSPACE

    #ifdef USE_TANGENT

      mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );

    #else

      mat3 tbn = getTangentFrame( - vViewPosition, normal, normalMapUv );

    #endif

    #if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )

      tbn[0] *= faceDirection;
      tbn[1] *= faceDirection;

    #endif

  #endif

  #ifdef USE_CLEARCOAT_NORMALMAP

    #ifdef USE_TANGENT

      mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );

    #else

      mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );

    #endif

    #if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )

      tbn2[0] *= faceDirection;
      tbn2[1] *= faceDirection;

    #endif

  #endif

  // non perturbed normal for clearcoat among others

  vec3 nonPerturbedNormal = normal;

  #ifdef OUTLINE
    normal *= -1.0;
  #endif

  // #include <normal_fragment_maps>

  // COMPAT: pre-r151
  // USE_NORMALMAP_OBJECTSPACE used to be OBJECTSPACE_NORMALMAP in pre-r151
  #if defined( USE_NORMALMAP_OBJECTSPACE ) || defined( OBJECTSPACE_NORMALMAP )

    normal = texture2D( normalMap, normalMapUv ).xyz * 2.0 - 1.0; // overrides both flatShading and attribute normals

    #ifdef FLIP_SIDED

      normal = - normal;

    #endif

    #ifdef DOUBLE_SIDED

      normal = normal * faceDirection;

    #endif

    normal = normalize( normalMatrix * normal );

  // COMPAT: pre-r151
  // USE_NORMALMAP_TANGENTSPACE used to be TANGENTSPACE_NORMALMAP in pre-r151
  #elif defined( USE_NORMALMAP_TANGENTSPACE ) || defined( TANGENTSPACE_NORMALMAP )

    vec3 mapN = texture2D( normalMap, normalMapUv ).xyz * 2.0 - 1.0;
    mapN.xy *= normalScale;

    // COMPAT: pre-r151
    #if THREE_VRM_THREE_REVISION >= 151 || defined( USE_TANGENT )

      normal = normalize( tbn * mapN );

    #else

      normal = perturbNormal2Arb( uv, -vViewPosition, normal, mapN, faceDirection );

    #endif

  #endif

  // #include <emissivemap_fragment>
  #ifdef USE_EMISSIVEMAP
    vec2 emissiveMapUv = ( emissiveMapUvTransform * vec3( uv, 1 ) ).xy;
    totalEmissiveRadiance *= texture2D( emissiveMap, emissiveMapUv ).rgb;
  #endif

  #ifdef DEBUG_NORMAL
    gl_FragColor = vec4( 0.5 + 0.5 * normal, 1.0 );
    return;
  #endif

  // -- MToon: lighting --------------------------------------------------------
  // accumulation
  // #include <lights_phong_fragment>
  MToonMaterial material;

  material.diffuseColor = diffuseColor.rgb;

  material.shadeColor = shadeColorFactor;
  #ifdef USE_SHADEMULTIPLYTEXTURE
    vec2 shadeMultiplyTextureUv = ( shadeMultiplyTextureUvTransform * vec3( uv, 1 ) ).xy;
    material.shadeColor *= texture2D( shadeMultiplyTexture, shadeMultiplyTextureUv ).rgb;
  #endif

  #if ( defined( USE_COLOR ) && !defined( IGNORE_VERTEX_COLOR ) )
    material.shadeColor.rgb *= vColor;
  #endif

  material.shadingShift = shadingShiftFactor;
  #ifdef USE_SHADINGSHIFTTEXTURE
    vec2 shadingShiftTextureUv = ( shadingShiftTextureUvTransform * vec3( uv, 1 ) ).xy;
    material.shadingShift += texture2D( shadingShiftTexture, shadingShiftTextureUv ).r * shadingShiftTextureScale;
  #endif

  // #include <lights_fragment_begin>

  // MToon Specific changes:
  // Since we want to take shadows into account of shading instead of irradiance,
  // we had to modify the codes that multiplies the results of shadowmap into color of direct lights.

  // COMPAT: pre-r156 uses a struct GeometricContext
  #if THREE_VRM_THREE_REVISION >= 157
    vec3 geometryPosition = - vViewPosition;
    vec3 geometryNormal = normal;
    vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );

    vec3 geometryClearcoatNormal;

    #ifdef USE_CLEARCOAT

      geometryClearcoatNormal = clearcoatNormal;

    #endif
  #else
    GeometricContext geometry;

    geometry.position = - vViewPosition;
    geometry.normal = normal;
    geometry.viewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );

    #ifdef USE_CLEARCOAT

      geometry.clearcoatNormal = clearcoatNormal;

    #endif
  #endif

  IncidentLight directLight;

  // since these variables will be used in unrolled loop, we have to define in prior
  float shadow;

  #if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )

    PointLight pointLight;
    #if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
    PointLightShadow pointLightShadow;
    #endif

    #pragma unroll_loop_start
    for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {

      pointLight = pointLights[ i ];

      // COMPAT: pre-r156 uses a struct GeometricContext
      #if THREE_VRM_THREE_REVISION >= 157
        getPointLightInfo( pointLight, geometryPosition, directLight );
      #else
        getPointLightInfo( pointLight, geometry, directLight );
      #endif

      shadow = 1.0;
      #if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS )
      pointLightShadow = pointLightShadows[ i ];
      // COMPAT: pre-r166
      // r166 introduced shadowIntensity
      #if THREE_VRM_THREE_REVISION >= 166
        shadow = all( bvec2( directLight.visible, receiveShadow ) ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
      #else
        shadow = all( bvec2( directLight.visible, receiveShadow ) ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
      #endif
      #endif

      // COMPAT: pre-r156 uses a struct GeometricContext
      #if THREE_VRM_THREE_REVISION >= 157
        RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, shadow, reflectedLight );
      #else
        RE_Direct( directLight, geometry, material, shadow, reflectedLight );
      #endif

    }
    #pragma unroll_loop_end

  #endif

  #if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )

    SpotLight spotLight;
    // COMPAT: pre-r144 uses NUM_SPOT_LIGHT_SHADOWS, r144+ uses NUM_SPOT_LIGHT_COORDS
    #if THREE_VRM_THREE_REVISION >= 144
      #if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_COORDS > 0
      SpotLightShadow spotLightShadow;
      #endif
    #elif defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
    SpotLightShadow spotLightShadow;
    #endif

    #pragma unroll_loop_start
    for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {

      spotLight = spotLights[ i ];

      // COMPAT: pre-r156 uses a struct GeometricContext
      #if THREE_VRM_THREE_REVISION >= 157
        getSpotLightInfo( spotLight, geometryPosition, directLight );
      #else
        getSpotLightInfo( spotLight, geometry, directLight );
      #endif

      shadow = 1.0;
      // COMPAT: pre-r144 uses NUM_SPOT_LIGHT_SHADOWS and vSpotShadowCoord, r144+ uses NUM_SPOT_LIGHT_COORDS and vSpotLightCoord
      // COMPAT: pre-r166 does not have shadowIntensity, r166+ has shadowIntensity
      #if THREE_VRM_THREE_REVISION >= 166
        #if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_COORDS )
        spotLightShadow = spotLightShadows[ i ];
        shadow = all( bvec2( directLight.visible, receiveShadow ) ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
        #endif
      #elif THREE_VRM_THREE_REVISION >= 144
        #if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_COORDS )
        spotLightShadow = spotLightShadows[ i ];
        shadow = all( bvec2( directLight.visible, receiveShadow ) ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
        #endif
      #elif defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
      spotLightShadow = spotLightShadows[ i ];
      shadow = all( bvec2( directLight.visible, receiveShadow ) ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotShadowCoord[ i ] ) : 1.0;
      #endif

      // COMPAT: pre-r156 uses a struct GeometricContext
      #if THREE_VRM_THREE_REVISION >= 157
        RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, shadow, reflectedLight );
      #else
        RE_Direct( directLight, geometry, material, shadow, reflectedLight );
      #endif

    }
    #pragma unroll_loop_end

  #endif

  #if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )

    DirectionalLight directionalLight;
    #if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
    DirectionalLightShadow directionalLightShadow;
    #endif

    #pragma unroll_loop_start
    for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {

      directionalLight = directionalLights[ i ];

      // COMPAT: pre-r156 uses a struct GeometricContext
      #if THREE_VRM_THREE_REVISION >= 157
        getDirectionalLightInfo( directionalLight, directLight );
      #else
        getDirectionalLightInfo( directionalLight, geometry, directLight );
      #endif

      shadow = 1.0;
      #if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
      directionalLightShadow = directionalLightShadows[ i ];
      // COMPAT: pre-r166
      // r166 introduced shadowIntensity
      #if THREE_VRM_THREE_REVISION >= 166
        shadow = all( bvec2( directLight.visible, receiveShadow ) ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
      #else
        shadow = all( bvec2( directLight.visible, receiveShadow ) ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
      #endif
      #endif

      // COMPAT: pre-r156 uses a struct GeometricContext
      #if THREE_VRM_THREE_REVISION >= 157
        RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, shadow, reflectedLight );
      #else
        RE_Direct( directLight, geometry, material, shadow, reflectedLight );
      #endif

    }
    #pragma unroll_loop_end

  #endif

  // #if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )

  //   RectAreaLight rectAreaLight;

  //   #pragma unroll_loop_start
  //   for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {

  //     rectAreaLight = rectAreaLights[ i ];
  //     RE_Direct_RectArea( rectAreaLight, geometry, material, reflectedLight );

  //   }
  //   #pragma unroll_loop_end

  // #endif

  #if defined( RE_IndirectDiffuse )

    vec3 iblIrradiance = vec3( 0.0 );

    vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );

    // COMPAT: pre-r156 uses a struct GeometricContext
    // COMPAT: pre-r156 doesn't have a define USE_LIGHT_PROBES
    #if THREE_VRM_THREE_REVISION >= 157
      #if defined( USE_LIGHT_PROBES )
        irradiance += getLightProbeIrradiance( lightProbe, geometryNormal );
      #endif
    #else
      irradiance += getLightProbeIrradiance( lightProbe, geometry.normal );
    #endif

    #if ( NUM_HEMI_LIGHTS > 0 )

      #pragma unroll_loop_start
      for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {

        // COMPAT: pre-r156 uses a struct GeometricContext
        #if THREE_VRM_THREE_REVISION >= 157
          irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );
        #else
          irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometry.normal );
        #endif

      }
      #pragma unroll_loop_end

    #endif

  #endif

  // #if defined( RE_IndirectSpecular )

  //   vec3 radiance = vec3( 0.0 );
  //   vec3 clearcoatRadiance = vec3( 0.0 );

  // #endif

  #include <lights_fragment_maps>
  #include <lights_fragment_end>

  // modulation
  #include <aomap_fragment>

  vec3 col = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;

  #ifdef DEBUG_LITSHADERATE
    gl_FragColor = vec4( col, diffuseColor.a );
    postCorrection();
    return;
  #endif

  // -- MToon: rim lighting -----------------------------------------
  vec3 viewDir = normalize( vViewPosition );

  #ifndef PHYSICALLY_CORRECT_LIGHTS
    reflectedLight.directSpecular /= PI;
  #endif
  vec3 rimMix = mix( vec3( 1.0 ), reflectedLight.directSpecular, rimLightingMixFactor );

  vec3 rim = parametricRimColorFactor * pow( saturate( 1.0 - dot( viewDir, normal ) + parametricRimLiftFactor ), parametricRimFresnelPowerFactor );

  #ifdef USE_MATCAPTEXTURE
    {
      vec3 x = normalize( vec3( viewDir.z, 0.0, -viewDir.x ) );
      vec3 y = cross( viewDir, x ); // guaranteed to be normalized
      vec2 sphereUv = 0.5 + 0.5 * vec2( dot( x, normal ), -dot( y, normal ) );
      sphereUv = ( matcapTextureUvTransform * vec3( sphereUv, 1 ) ).xy;
      vec3 matcap = texture2D( matcapTexture, sphereUv ).rgb;
      rim += matcapFactor * matcap;
    }
  #endif

  #ifdef USE_RIMMULTIPLYTEXTURE
    vec2 rimMultiplyTextureUv = ( rimMultiplyTextureUvTransform * vec3( uv, 1 ) ).xy;
    rim *= texture2D( rimMultiplyTexture, rimMultiplyTextureUv ).rgb;
  #endif

  col += rimMix * rim;

  // -- MToon: Emission --------------------------------------------------------
  col += totalEmissiveRadiance;

  // #include <envmap_fragment>

  // -- Almost done! -----------------------------------------------------------
  #if defined( OUTLINE )
    col = outlineColorFactor.rgb * mix( vec3( 1.0 ), col, outlineLightingMixFactor );
  #endif

  #ifdef OPAQUE
    diffuseColor.a = 1.0;
  #endif

  gl_FragColor = vec4( col, diffuseColor.a );
  postCorrection();
}
`,Xo={None:"none"},Nn={None:"none",ScreenCoordinates:"screenCoordinates"},qo={3e3:"",3001:"srgb"};function nt(e){return parseInt(We,10)>=152?e.colorSpace:qo[e.encoding]}var Yo=class extends pr{constructor(e={}){var t;super({vertexShader:jo,fragmentShader:Go}),this.uvAnimationScrollXSpeedFactor=0,this.uvAnimationScrollYSpeedFactor=0,this.uvAnimationRotationSpeedFactor=0,this.fog=!0,this.normalMapType=fr,this._ignoreVertexColor=!0,this._v0CompatShade=!1,this._debugMode=Xo.None,this._outlineWidthMode=Nn.None,this._isOutline=!1,e.transparentWithZWrite&&(e.depthWrite=!0),delete e.transparentWithZWrite,e.fog=!0,e.lights=!0,e.clipping=!0,this.uniforms=mr.merge([xe.common,xe.normalmap,xe.emissivemap,xe.fog,xe.lights,{litFactor:{value:new H(1,1,1)},mapUvTransform:{value:new $},colorAlpha:{value:1},normalMapUvTransform:{value:new $},shadeColorFactor:{value:new H(0,0,0)},shadeMultiplyTexture:{value:null},shadeMultiplyTextureUvTransform:{value:new $},shadingShiftFactor:{value:0},shadingShiftTexture:{value:null},shadingShiftTextureUvTransform:{value:new $},shadingShiftTextureScale:{value:1},shadingToonyFactor:{value:.9},giEqualizationFactor:{value:.9},matcapFactor:{value:new H(1,1,1)},matcapTexture:{value:null},matcapTextureUvTransform:{value:new $},parametricRimColorFactor:{value:new H(0,0,0)},rimMultiplyTexture:{value:null},rimMultiplyTextureUvTransform:{value:new $},rimLightingMixFactor:{value:1},parametricRimFresnelPowerFactor:{value:5},parametricRimLiftFactor:{value:0},emissive:{value:new H(0,0,0)},emissiveIntensity:{value:1},emissiveMapUvTransform:{value:new $},outlineWidthMultiplyTexture:{value:null},outlineWidthMultiplyTextureUvTransform:{value:new $},outlineWidthFactor:{value:0},outlineColorFactor:{value:new H(0,0,0)},outlineLightingMixFactor:{value:1},uvAnimationMaskTexture:{value:null},uvAnimationMaskTextureUvTransform:{value:new $},uvAnimationScrollXOffset:{value:0},uvAnimationScrollYOffset:{value:0},uvAnimationRotationPhase:{value:0}},(t=e.uniforms)!=null?t:{}]),this.setValues(e),this._uploadUniformsWorkaround(),this.customProgramCacheKey=()=>[...Object.entries(this._generateDefines()).map(([n,i])=>`${n}:${i}`),this.matcapTexture?`matcapTextureColorSpace:${nt(this.matcapTexture)}`:"",this.shadeMultiplyTexture?`shadeMultiplyTextureColorSpace:${nt(this.shadeMultiplyTexture)}`:"",this.rimMultiplyTexture?`rimMultiplyTextureColorSpace:${nt(this.rimMultiplyTexture)}`:""].join(","),this.onBeforeCompile=n=>{const i=parseInt(We,10),r=Object.entries(Cn(Cn({},this._generateDefines()),this.defines)).filter(([o,a])=>!!a).map(([o,a])=>`#define ${o} ${a}`).join(`
`)+`
`;n.vertexShader=r+n.vertexShader,n.fragmentShader=r+n.fragmentShader,i<154&&(n.fragmentShader=n.fragmentShader.replace("#include <colorspace_fragment>","#include <encodings_fragment>"))}}get color(){return this.uniforms.litFactor.value}set color(e){this.uniforms.litFactor.value=e}get map(){return this.uniforms.map.value}set map(e){this.uniforms.map.value=e}get normalMap(){return this.uniforms.normalMap.value}set normalMap(e){this.uniforms.normalMap.value=e}get normalScale(){return this.uniforms.normalScale.value}set normalScale(e){this.uniforms.normalScale.value=e}get emissive(){return this.uniforms.emissive.value}set emissive(e){this.uniforms.emissive.value=e}get emissiveIntensity(){return this.uniforms.emissiveIntensity.value}set emissiveIntensity(e){this.uniforms.emissiveIntensity.value=e}get emissiveMap(){return this.uniforms.emissiveMap.value}set emissiveMap(e){this.uniforms.emissiveMap.value=e}get shadeColorFactor(){return this.uniforms.shadeColorFactor.value}set shadeColorFactor(e){this.uniforms.shadeColorFactor.value=e}get shadeMultiplyTexture(){return this.uniforms.shadeMultiplyTexture.value}set shadeMultiplyTexture(e){this.uniforms.shadeMultiplyTexture.value=e}get shadingShiftFactor(){return this.uniforms.shadingShiftFactor.value}set shadingShiftFactor(e){this.uniforms.shadingShiftFactor.value=e}get shadingShiftTexture(){return this.uniforms.shadingShiftTexture.value}set shadingShiftTexture(e){this.uniforms.shadingShiftTexture.value=e}get shadingShiftTextureScale(){return this.uniforms.shadingShiftTextureScale.value}set shadingShiftTextureScale(e){this.uniforms.shadingShiftTextureScale.value=e}get shadingToonyFactor(){return this.uniforms.shadingToonyFactor.value}set shadingToonyFactor(e){this.uniforms.shadingToonyFactor.value=e}get giEqualizationFactor(){return this.uniforms.giEqualizationFactor.value}set giEqualizationFactor(e){this.uniforms.giEqualizationFactor.value=e}get matcapFactor(){return this.uniforms.matcapFactor.value}set matcapFactor(e){this.uniforms.matcapFactor.value=e}get matcapTexture(){return this.uniforms.matcapTexture.value}set matcapTexture(e){this.uniforms.matcapTexture.value=e}get parametricRimColorFactor(){return this.uniforms.parametricRimColorFactor.value}set parametricRimColorFactor(e){this.uniforms.parametricRimColorFactor.value=e}get rimMultiplyTexture(){return this.uniforms.rimMultiplyTexture.value}set rimMultiplyTexture(e){this.uniforms.rimMultiplyTexture.value=e}get rimLightingMixFactor(){return this.uniforms.rimLightingMixFactor.value}set rimLightingMixFactor(e){this.uniforms.rimLightingMixFactor.value=e}get parametricRimFresnelPowerFactor(){return this.uniforms.parametricRimFresnelPowerFactor.value}set parametricRimFresnelPowerFactor(e){this.uniforms.parametricRimFresnelPowerFactor.value=e}get parametricRimLiftFactor(){return this.uniforms.parametricRimLiftFactor.value}set parametricRimLiftFactor(e){this.uniforms.parametricRimLiftFactor.value=e}get outlineWidthMultiplyTexture(){return this.uniforms.outlineWidthMultiplyTexture.value}set outlineWidthMultiplyTexture(e){this.uniforms.outlineWidthMultiplyTexture.value=e}get outlineWidthFactor(){return this.uniforms.outlineWidthFactor.value}set outlineWidthFactor(e){this.uniforms.outlineWidthFactor.value=e}get outlineColorFactor(){return this.uniforms.outlineColorFactor.value}set outlineColorFactor(e){this.uniforms.outlineColorFactor.value=e}get outlineLightingMixFactor(){return this.uniforms.outlineLightingMixFactor.value}set outlineLightingMixFactor(e){this.uniforms.outlineLightingMixFactor.value=e}get uvAnimationMaskTexture(){return this.uniforms.uvAnimationMaskTexture.value}set uvAnimationMaskTexture(e){this.uniforms.uvAnimationMaskTexture.value=e}get uvAnimationScrollXOffset(){return this.uniforms.uvAnimationScrollXOffset.value}set uvAnimationScrollXOffset(e){this.uniforms.uvAnimationScrollXOffset.value=e}get uvAnimationScrollYOffset(){return this.uniforms.uvAnimationScrollYOffset.value}set uvAnimationScrollYOffset(e){this.uniforms.uvAnimationScrollYOffset.value=e}get uvAnimationRotationPhase(){return this.uniforms.uvAnimationRotationPhase.value}set uvAnimationRotationPhase(e){this.uniforms.uvAnimationRotationPhase.value=e}get ignoreVertexColor(){return this._ignoreVertexColor}set ignoreVertexColor(e){this._ignoreVertexColor=e,this.needsUpdate=!0}get v0CompatShade(){return this._v0CompatShade}set v0CompatShade(e){this._v0CompatShade=e,this.needsUpdate=!0}get debugMode(){return this._debugMode}set debugMode(e){this._debugMode=e,this.needsUpdate=!0}get outlineWidthMode(){return this._outlineWidthMode}set outlineWidthMode(e){this._outlineWidthMode=e,this.needsUpdate=!0}get isOutline(){return this._isOutline}set isOutline(e){this._isOutline=e,this.needsUpdate=!0}get isMToonMaterial(){return!0}update(e){this._uploadUniformsWorkaround(),this._updateUVAnimation(e)}copy(e){return super.copy(e),this.map=e.map,this.normalMap=e.normalMap,this.emissiveMap=e.emissiveMap,this.shadeMultiplyTexture=e.shadeMultiplyTexture,this.shadingShiftTexture=e.shadingShiftTexture,this.matcapTexture=e.matcapTexture,this.rimMultiplyTexture=e.rimMultiplyTexture,this.outlineWidthMultiplyTexture=e.outlineWidthMultiplyTexture,this.uvAnimationMaskTexture=e.uvAnimationMaskTexture,this.normalMapType=e.normalMapType,this.uvAnimationScrollXSpeedFactor=e.uvAnimationScrollXSpeedFactor,this.uvAnimationScrollYSpeedFactor=e.uvAnimationScrollYSpeedFactor,this.uvAnimationRotationSpeedFactor=e.uvAnimationRotationSpeedFactor,this.ignoreVertexColor=e.ignoreVertexColor,this.v0CompatShade=e.v0CompatShade,this.debugMode=e.debugMode,this.outlineWidthMode=e.outlineWidthMode,this.isOutline=e.isOutline,this.needsUpdate=!0,this}_updateUVAnimation(e){this.uniforms.uvAnimationScrollXOffset.value+=e*this.uvAnimationScrollXSpeedFactor,this.uniforms.uvAnimationScrollYOffset.value+=e*this.uvAnimationScrollYSpeedFactor,this.uniforms.uvAnimationRotationPhase.value+=e*this.uvAnimationRotationSpeedFactor,this.uniforms.alphaTest.value=this.alphaTest,this.uniformsNeedUpdate=!0}_uploadUniformsWorkaround(){this.uniforms.opacity.value=this.opacity,this._updateTextureMatrix(this.uniforms.map,this.uniforms.mapUvTransform),this._updateTextureMatrix(this.uniforms.normalMap,this.uniforms.normalMapUvTransform),this._updateTextureMatrix(this.uniforms.emissiveMap,this.uniforms.emissiveMapUvTransform),this._updateTextureMatrix(this.uniforms.shadeMultiplyTexture,this.uniforms.shadeMultiplyTextureUvTransform),this._updateTextureMatrix(this.uniforms.shadingShiftTexture,this.uniforms.shadingShiftTextureUvTransform),this._updateTextureMatrix(this.uniforms.matcapTexture,this.uniforms.matcapTextureUvTransform),this._updateTextureMatrix(this.uniforms.rimMultiplyTexture,this.uniforms.rimMultiplyTextureUvTransform),this._updateTextureMatrix(this.uniforms.outlineWidthMultiplyTexture,this.uniforms.outlineWidthMultiplyTextureUvTransform),this._updateTextureMatrix(this.uniforms.uvAnimationMaskTexture,this.uniforms.uvAnimationMaskTextureUvTransform),this.uniformsNeedUpdate=!0}_generateDefines(){const e=parseInt(We,10),t=this.outlineWidthMultiplyTexture!==null,n=this.map!==null||this.normalMap!==null||this.emissiveMap!==null||this.shadeMultiplyTexture!==null||this.shadingShiftTexture!==null||this.rimMultiplyTexture!==null||this.uvAnimationMaskTexture!==null;return{THREE_VRM_THREE_REVISION:e,OUTLINE:this._isOutline,MTOON_USE_UV:t||n,MTOON_UVS_VERTEX_ONLY:t&&!n,V0_COMPAT_SHADE:this._v0CompatShade,USE_SHADEMULTIPLYTEXTURE:this.shadeMultiplyTexture!==null,USE_SHADINGSHIFTTEXTURE:this.shadingShiftTexture!==null,USE_MATCAPTEXTURE:this.matcapTexture!==null,USE_RIMMULTIPLYTEXTURE:this.rimMultiplyTexture!==null,USE_OUTLINEWIDTHMULTIPLYTEXTURE:this._isOutline&&this.outlineWidthMultiplyTexture!==null,USE_UVANIMATIONMASKTEXTURE:this.uvAnimationMaskTexture!==null,IGNORE_VERTEX_COLOR:this._ignoreVertexColor===!0,DEBUG_NORMAL:this._debugMode==="normal",DEBUG_LITSHADERATE:this._debugMode==="litShadeRate",DEBUG_UV:this._debugMode==="uv",OUTLINE_WIDTH_SCREEN:this._isOutline&&this._outlineWidthMode===Nn.ScreenCoordinates}}_updateTextureMatrix(e,t){e.value&&(e.value.matrixAutoUpdate&&e.value.updateMatrix(),t.value.copy(e.value.matrix))}},Qo=new Set(["1.0","1.0-beta"]),Ti=class He{get name(){return He.EXTENSION_NAME}constructor(t,n={}){var i,r,o,a;this.parser=t,this.materialType=(i=n.materialType)!=null?i:Yo,this.renderOrderOffset=(r=n.renderOrderOffset)!=null?r:0,this.v0CompatShade=(o=n.v0CompatShade)!=null?o:!1,this.debugMode=(a=n.debugMode)!=null?a:"none",this._mToonMaterialSet=new Set}beforeRoot(){return le(this,null,function*(){this._removeUnlitExtensionIfMToonExists()})}afterRoot(t){return le(this,null,function*(){t.userData.vrmMToonMaterials=Array.from(this._mToonMaterialSet)})}getMaterialType(t){return this._getMToonExtension(t)?this.materialType:null}extendMaterialParams(t,n){const i=this._getMToonExtension(t);return i?this._extendMaterialParams(i,n):null}loadMesh(t){return le(this,null,function*(){var n;const i=this.parser,o=(n=i.json.meshes)==null?void 0:n[t];if(o==null)throw new Error(`MToonMaterialLoaderPlugin: Attempt to use meshes[${t}] of glTF but the mesh doesn't exist`);const a=o.primitives,u=yield i.loadMesh(t);if(a.length===1){const s=u,l=a[0].material;l!=null&&this._setupPrimitive(s,l)}else{const s=u;for(let l=0;l<a.length;l++){const d=s.children[l],c=a[l].material;c!=null&&this._setupPrimitive(d,c)}}return u})}_removeUnlitExtensionIfMToonExists(){const i=this.parser.json.materials;i==null||i.map((r,o)=>{var a;this._getMToonExtension(o)&&((a=r.extensions)!=null&&a.KHR_materials_unlit)&&delete r.extensions.KHR_materials_unlit})}_getMToonExtension(t){var n,i;const a=(n=this.parser.json.materials)==null?void 0:n[t];if(a==null){console.warn(`MToonMaterialLoaderPlugin: Attempt to use materials[${t}] of glTF but the material doesn't exist`);return}const u=(i=a.extensions)==null?void 0:i[He.EXTENSION_NAME];if(u==null)return;const s=u.specVersion;if(!Qo.has(s)){console.warn(`MToonMaterialLoaderPlugin: Unknown ${He.EXTENSION_NAME} specVersion "${s}"`);return}return u}_extendMaterialParams(t,n){return le(this,null,function*(){var i;delete n.metalness,delete n.roughness;const r=new zo(this.parser,n);r.assignPrimitive("transparentWithZWrite",t.transparentWithZWrite),r.assignColor("shadeColorFactor",t.shadeColorFactor),r.assignTexture("shadeMultiplyTexture",t.shadeMultiplyTexture,!0),r.assignPrimitive("shadingShiftFactor",t.shadingShiftFactor),r.assignTexture("shadingShiftTexture",t.shadingShiftTexture,!0),r.assignPrimitive("shadingShiftTextureScale",(i=t.shadingShiftTexture)==null?void 0:i.scale),r.assignPrimitive("shadingToonyFactor",t.shadingToonyFactor),r.assignPrimitive("giEqualizationFactor",t.giEqualizationFactor),r.assignColor("matcapFactor",t.matcapFactor),r.assignTexture("matcapTexture",t.matcapTexture,!0),r.assignColor("parametricRimColorFactor",t.parametricRimColorFactor),r.assignTexture("rimMultiplyTexture",t.rimMultiplyTexture,!0),r.assignPrimitive("rimLightingMixFactor",t.rimLightingMixFactor),r.assignPrimitive("parametricRimFresnelPowerFactor",t.parametricRimFresnelPowerFactor),r.assignPrimitive("parametricRimLiftFactor",t.parametricRimLiftFactor),r.assignPrimitive("outlineWidthMode",t.outlineWidthMode),r.assignPrimitive("outlineWidthFactor",t.outlineWidthFactor),r.assignTexture("outlineWidthMultiplyTexture",t.outlineWidthMultiplyTexture,!1),r.assignColor("outlineColorFactor",t.outlineColorFactor),r.assignPrimitive("outlineLightingMixFactor",t.outlineLightingMixFactor),r.assignTexture("uvAnimationMaskTexture",t.uvAnimationMaskTexture,!1),r.assignPrimitive("uvAnimationScrollXSpeedFactor",t.uvAnimationScrollXSpeedFactor),r.assignPrimitive("uvAnimationScrollYSpeedFactor",t.uvAnimationScrollYSpeedFactor),r.assignPrimitive("uvAnimationRotationSpeedFactor",t.uvAnimationRotationSpeedFactor),r.assignPrimitive("v0CompatShade",this.v0CompatShade),r.assignPrimitive("debugMode",this.debugMode),yield r.pending})}_setupPrimitive(t,n){const i=this._getMToonExtension(n);if(i){const r=this._parseRenderOrder(i);t.renderOrder=r+this.renderOrderOffset,this._generateOutline(t),this._addToMaterialSet(t);return}}_shouldGenerateOutline(t){return typeof t.outlineWidthMode=="string"&&t.outlineWidthMode!=="none"&&typeof t.outlineWidthFactor=="number"&&t.outlineWidthFactor>0}_generateOutline(t){const n=t.material;if(!(n instanceof dr)||!this._shouldGenerateOutline(n))return;t.material=[n];const i=n.clone();i.name+=" (Outline)",i.isOutline=!0,i.side=cr,t.material.push(i);const r=t.geometry,o=r.index?r.index.count:r.attributes.position.count/3;r.addGroup(0,o,0),r.addGroup(0,o,1)}_addToMaterialSet(t){const n=t.material,i=new Set;Array.isArray(n)?n.forEach(r=>i.add(r)):i.add(n);for(const r of i)this._mToonMaterialSet.add(r)}_parseRenderOrder(t){var n;return(t.transparentWithZWrite?0:19)+((n=t.renderQueueOffsetNumber)!=null?n:0)}};Ti.EXTENSION_NAME="VRMC_materials_mtoon";var $o=Ti,Zo=(e,t,n)=>new Promise((i,r)=>{var o=s=>{try{u(n.next(s))}catch(l){r(l)}},a=s=>{try{u(n.throw(s))}catch(l){r(l)}},u=s=>s.done?i(s.value):Promise.resolve(s.value).then(o,a);u((n=n.apply(e,t)).next())}),Ri=class Mt{get name(){return Mt.EXTENSION_NAME}constructor(t){this.parser=t}extendMaterialParams(t,n){return Zo(this,null,function*(){const i=this._getHDREmissiveMultiplierExtension(t);if(i==null)return;console.warn("VRMMaterialsHDREmissiveMultiplierLoaderPlugin: `VRMC_materials_hdr_emissiveMultiplier` is archived. Use `KHR_materials_emissive_strength` instead.");const r=i.emissiveMultiplier;n.emissiveIntensity=r})}_getHDREmissiveMultiplierExtension(t){var n,i;const a=(n=this.parser.json.materials)==null?void 0:n[t];if(a==null){console.warn(`VRMMaterialsHDREmissiveMultiplierLoaderPlugin: Attempt to use materials[${t}] of glTF but the material doesn't exist`);return}const u=(i=a.extensions)==null?void 0:i[Mt.EXTENSION_NAME];if(u!=null)return u}};Ri.EXTENSION_NAME="VRMC_materials_hdr_emissiveMultiplier";var Jo=Ri,Ko=Object.defineProperty,es=Object.defineProperties,ts=Object.getOwnPropertyDescriptors,On=Object.getOwnPropertySymbols,ns=Object.prototype.hasOwnProperty,is=Object.prototype.propertyIsEnumerable,Un=(e,t,n)=>t in e?Ko(e,t,{enumerable:!0,configurable:!0,writable:!0,value:n}):e[t]=n,Y=(e,t)=>{for(var n in t||(t={}))ns.call(t,n)&&Un(e,n,t[n]);if(On)for(var n of On(t))is.call(t,n)&&Un(e,n,t[n]);return e},Vn=(e,t)=>es(e,ts(t)),rs=(e,t,n)=>new Promise((i,r)=>{var o=s=>{try{u(n.next(s))}catch(l){r(l)}},a=s=>{try{u(n.throw(s))}catch(l){r(l)}},u=s=>s.done?i(s.value):Promise.resolve(s.value).then(o,a);u((n=n.apply(e,t)).next())});function ce(e){return Math.pow(e,2.2)}var os=class{get name(){return"VRMMaterialsV0CompatPlugin"}constructor(e){var t;this.parser=e,this._renderQueueMapTransparent=new Map,this._renderQueueMapTransparentZWrite=new Map;const n=this.parser.json;n.extensionsUsed=(t=n.extensionsUsed)!=null?t:[],n.extensionsUsed.indexOf("KHR_texture_transform")===-1&&n.extensionsUsed.push("KHR_texture_transform")}beforeRoot(){return rs(this,null,function*(){var e;const t=this.parser.json,n=(e=t.extensions)==null?void 0:e.VRM,i=n==null?void 0:n.materialProperties;i&&(this._populateRenderQueueMap(i),i.forEach((r,o)=>{var a,u;const s=(a=t.materials)==null?void 0:a[o];if(s==null){console.warn(`VRMMaterialsV0CompatPlugin: Attempt to use materials[${o}] of glTF but the material doesn't exist`);return}if(r.shader==="VRM/MToon"){const l=this._parseV0MToonProperties(r,s);t.materials[o]=l}else if((u=r.shader)!=null&&u.startsWith("VRM/Unlit")){const l=this._parseV0UnlitProperties(r,s);t.materials[o]=l}else r.shader==="VRM_USE_GLTFSHADER"||console.warn(`VRMMaterialsV0CompatPlugin: Unknown shader: ${r.shader}`)}))})}_parseV0MToonProperties(e,t){var n,i,r,o,a,u,s,l,d,c,p,h,_,g,m,f,x,T,y,w,R,A,P,N,b,C,D,W,G,M,L,B,z,X,V,K,te,k,Le,be,Ie,Me,Pt,Lt,bt,It,Ct,Nt,Ot,Ut,Vt,kt,Dt,Bt,Ft;const Ht=(i=(n=e.keywordMap)==null?void 0:n._ALPHABLEND_ON)!=null?i:!1,Oi=((r=e.floatProperties)==null?void 0:r._ZWrite)===1&&Ht,Ui=this._v0ParseRenderQueue(e),Wt=(a=(o=e.keywordMap)==null?void 0:o._ALPHATEST_ON)!=null?a:!1,Vi=Ht?"BLEND":Wt?"MASK":"OPAQUE",ki=Wt?(s=(u=e.floatProperties)==null?void 0:u._Cutoff)!=null?s:.5:void 0,Di=((d=(l=e.floatProperties)==null?void 0:l._CullMode)!=null?d:2)===0,re=this._portTextureTransform(e),Bi=((p=(c=e.vectorProperties)==null?void 0:c._Color)!=null?p:[1,1,1,1]).map((Zt,ur)=>ur===3?Zt:ce(Zt)),zt=(h=e.textureProperties)==null?void 0:h._MainTex,Fi=zt!=null?{index:zt,extensions:Y({},re)}:void 0,Hi=(g=(_=e.floatProperties)==null?void 0:_._BumpScale)!=null?g:1,jt=(m=e.textureProperties)==null?void 0:m._BumpMap,Wi=jt!=null?{index:jt,scale:Hi,extensions:Y({},re)}:void 0,zi=((x=(f=e.vectorProperties)==null?void 0:f._EmissionColor)!=null?x:[0,0,0,1]).map(ce),Gt=(T=e.textureProperties)==null?void 0:T._EmissionMap,ji=Gt!=null?{index:Gt,extensions:Y({},re)}:void 0,Gi=((w=(y=e.vectorProperties)==null?void 0:y._ShadeColor)!=null?w:[.97,.81,.86,1]).map(ce),Xt=(R=e.textureProperties)==null?void 0:R._ShadeTexture,Xi=Xt!=null?{index:Xt,extensions:Y({},re)}:void 0;let Ce=(P=(A=e.floatProperties)==null?void 0:A._ShadeShift)!=null?P:0,Ne=(b=(N=e.floatProperties)==null?void 0:N._ShadeToony)!=null?b:.9;Ne=O.lerp(Ne,1,.5+.5*Ce),Ce=-Ce-(1-Ne);const qt=(D=(C=e.floatProperties)==null?void 0:C._IndirectLightIntensity)!=null?D:.1,qi=qt?1-qt:void 0,Ye=(W=e.textureProperties)==null?void 0:W._SphereAdd,Yi=Ye!=null?[1,1,1]:void 0,Qi=Ye!=null?{index:Ye}:void 0,$i=(M=(G=e.floatProperties)==null?void 0:G._RimLightingMix)!=null?M:0,Yt=(L=e.textureProperties)==null?void 0:L._RimTexture,Zi=Yt!=null?{index:Yt,extensions:Y({},re)}:void 0,Ji=((z=(B=e.vectorProperties)==null?void 0:B._RimColor)!=null?z:[0,0,0,1]).map(ce),Ki=(V=(X=e.floatProperties)==null?void 0:X._RimFresnelPower)!=null?V:1,er=(te=(K=e.floatProperties)==null?void 0:K._RimLift)!=null?te:0,tr=["none","worldCoordinates","screenCoordinates"][(Le=(k=e.floatProperties)==null?void 0:k._OutlineWidthMode)!=null?Le:0];let Qe=(Ie=(be=e.floatProperties)==null?void 0:be._OutlineWidth)!=null?Ie:0;Qe=.01*Qe;const Qt=(Me=e.textureProperties)==null?void 0:Me._OutlineWidthTexture,nr=Qt!=null?{index:Qt,extensions:Y({},re)}:void 0,ir=((Lt=(Pt=e.vectorProperties)==null?void 0:Pt._OutlineColor)!=null?Lt:[0,0,0]).map(ce),rr=((It=(bt=e.floatProperties)==null?void 0:bt._OutlineColorMode)!=null?It:0)===1?(Nt=(Ct=e.floatProperties)==null?void 0:Ct._OutlineLightingMix)!=null?Nt:1:0,$t=(Ot=e.textureProperties)==null?void 0:Ot._UvAnimMaskTexture,or=$t!=null?{index:$t,extensions:Y({},re)}:void 0,sr=(Vt=(Ut=e.floatProperties)==null?void 0:Ut._UvAnimScrollX)!=null?Vt:0;let Oe=(Dt=(kt=e.floatProperties)==null?void 0:kt._UvAnimScrollY)!=null?Dt:0;Oe!=null&&(Oe=-Oe);const ar=(Ft=(Bt=e.floatProperties)==null?void 0:Bt._UvAnimRotation)!=null?Ft:0,lr={specVersion:"1.0",transparentWithZWrite:Oi,renderQueueOffsetNumber:Ui,shadeColorFactor:Gi,shadeMultiplyTexture:Xi,shadingShiftFactor:Ce,shadingToonyFactor:Ne,giEqualizationFactor:qi,matcapFactor:Yi,matcapTexture:Qi,rimLightingMixFactor:$i,rimMultiplyTexture:Zi,parametricRimColorFactor:Ji,parametricRimFresnelPowerFactor:Ki,parametricRimLiftFactor:er,outlineWidthMode:tr,outlineWidthFactor:Qe,outlineWidthMultiplyTexture:nr,outlineColorFactor:ir,outlineLightingMixFactor:rr,uvAnimationMaskTexture:or,uvAnimationScrollXSpeedFactor:sr,uvAnimationScrollYSpeedFactor:Oe,uvAnimationRotationSpeedFactor:ar};return Vn(Y({},t),{pbrMetallicRoughness:{baseColorFactor:Bi,baseColorTexture:Fi},normalTexture:Wi,emissiveTexture:ji,emissiveFactor:zi,alphaMode:Vi,alphaCutoff:ki,doubleSided:Di,extensions:{VRMC_materials_mtoon:lr}})}_parseV0UnlitProperties(e,t){var n,i,r,o,a;const u=e.shader==="VRM/UnlitTransparentZWrite",s=e.shader==="VRM/UnlitTransparent"||u,l=this._v0ParseRenderQueue(e),d=e.shader==="VRM/UnlitCutout",c=s?"BLEND":d?"MASK":"OPAQUE",p=d?(i=(n=e.floatProperties)==null?void 0:n._Cutoff)!=null?i:.5:void 0,h=this._portTextureTransform(e),_=((o=(r=e.vectorProperties)==null?void 0:r._Color)!=null?o:[1,1,1,1]).map(ce),g=(a=e.textureProperties)==null?void 0:a._MainTex,m=g!=null?{index:g,extensions:Y({},h)}:void 0,f={specVersion:"1.0",transparentWithZWrite:u,renderQueueOffsetNumber:l,shadeColorFactor:_,shadeMultiplyTexture:m};return Vn(Y({},t),{pbrMetallicRoughness:{baseColorFactor:_,baseColorTexture:m},alphaMode:c,alphaCutoff:p,extensions:{VRMC_materials_mtoon:f}})}_portTextureTransform(e){var t,n,i,r,o;const a=(t=e.vectorProperties)==null?void 0:t._MainTex;if(a==null)return{};const u=[(n=a==null?void 0:a[0])!=null?n:0,(i=a==null?void 0:a[1])!=null?i:0],s=[(r=a==null?void 0:a[2])!=null?r:1,(o=a==null?void 0:a[3])!=null?o:1];return u[1]=1-s[1]-u[1],{KHR_texture_transform:{offset:u,scale:s}}}_v0ParseRenderQueue(e){var t,n;const i=e.shader==="VRM/UnlitTransparentZWrite",r=((t=e.keywordMap)==null?void 0:t._ALPHABLEND_ON)!=null||e.shader==="VRM/UnlitTransparent"||i,o=((n=e.floatProperties)==null?void 0:n._ZWrite)===1||i;let a=0;if(r){const u=e.renderQueue;u!=null&&(o?a=this._renderQueueMapTransparentZWrite.get(u):a=this._renderQueueMapTransparent.get(u))}return a}_populateRenderQueueMap(e){const t=new Set,n=new Set;e.forEach(i=>{var r,o;const a=i.shader==="VRM/UnlitTransparentZWrite",u=((r=i.keywordMap)==null?void 0:r._ALPHABLEND_ON)!=null||i.shader==="VRM/UnlitTransparent"||a,s=((o=i.floatProperties)==null?void 0:o._ZWrite)===1||a;if(u){const l=i.renderQueue;l!=null&&(s?n.add(l):t.add(l))}}),t.size>10&&console.warn(`VRMMaterialsV0CompatPlugin: This VRM uses ${t.size} render queues for Transparent materials while VRM 1.0 only supports up to 10 render queues. The model might not be rendered correctly.`),n.size>10&&console.warn(`VRMMaterialsV0CompatPlugin: This VRM uses ${n.size} render queues for TransparentZWrite materials while VRM 1.0 only supports up to 10 render queues. The model might not be rendered correctly.`),Array.from(t).sort().forEach((i,r)=>{const o=Math.min(Math.max(r-t.size+1,-9),0);this._renderQueueMapTransparent.set(i,o)}),Array.from(n).sort().forEach((i,r)=>{const o=Math.min(Math.max(r,0),9);this._renderQueueMapTransparentZWrite.set(i,o)})}},kn=(e,t,n)=>new Promise((i,r)=>{var o=s=>{try{u(n.next(s))}catch(l){r(l)}},a=s=>{try{u(n.throw(s))}catch(l){r(l)}},u=s=>s.done?i(s.value):Promise.resolve(s.value).then(o,a);u((n=n.apply(e,t)).next())}),ne=new v,it=class extends ve{constructor(e){super(),this._attrPosition=new F(new Float32Array([0,0,0,0,0,0]),3),this._attrPosition.setUsage(_r);const t=new ee;t.setAttribute("position",this._attrPosition);const n=new qe({color:16711935,depthTest:!1,depthWrite:!1});this._line=new gr(t,n),this.add(this._line),this.constraint=e}updateMatrixWorld(e){ne.setFromMatrixPosition(this.constraint.destination.matrixWorld),this._attrPosition.setXYZ(0,ne.x,ne.y,ne.z),this.constraint.source&&ne.setFromMatrixPosition(this.constraint.source.matrixWorld),this._attrPosition.setXYZ(1,ne.x,ne.y,ne.z),this._attrPosition.needsUpdate=!0,super.updateMatrixWorld(e)}};function Dn(e,t){return t.set(e.elements[12],e.elements[13],e.elements[14])}var ss=new v,as=new v;function ls(e,t){return e.decompose(ss,t,as),t}function je(e){return e.invert?e.invert():e.inverse(),e}var At=class{constructor(e,t){this.destination=e,this.source=t,this.weight=1}},us=new v,ds=new v,cs=new v,hs=new S,ps=new S,fs=new S,ms=class extends At{get aimAxis(){return this._aimAxis}set aimAxis(e){this._aimAxis=e,this._v3AimAxis.set(e==="PositiveX"?1:e==="NegativeX"?-1:0,e==="PositiveY"?1:e==="NegativeY"?-1:0,e==="PositiveZ"?1:e==="NegativeZ"?-1:0)}get dependencies(){const e=new Set([this.source]);return this.destination.parent&&e.add(this.destination.parent),e}constructor(e,t){super(e,t),this._aimAxis="PositiveX",this._v3AimAxis=new v(1,0,0),this._dstRestQuat=new S}setInitState(){this._dstRestQuat.copy(this.destination.quaternion)}update(){this.destination.updateWorldMatrix(!0,!1),this.source.updateWorldMatrix(!0,!1);const e=hs.identity(),t=ps.identity();this.destination.parent&&(ls(this.destination.parent.matrixWorld,e),je(t.copy(e)));const n=us.copy(this._v3AimAxis).applyQuaternion(this._dstRestQuat).applyQuaternion(e),i=Dn(this.source.matrixWorld,ds).sub(Dn(this.destination.matrixWorld,cs)).normalize(),r=fs.setFromUnitVectors(n,i).premultiply(t).multiply(e).multiply(this._dstRestQuat);this.destination.quaternion.copy(this._dstRestQuat).slerp(r,this.weight)}};function _s(e,t){const n=[e];let i=e.parent;for(;i!==null;)n.unshift(i),i=i.parent;n.forEach(r=>{t(r)})}var gs=class{constructor(){this._constraints=new Set,this._objectConstraintsMap=new Map}get constraints(){return this._constraints}addConstraint(e){this._constraints.add(e);let t=this._objectConstraintsMap.get(e.destination);t==null&&(t=new Set,this._objectConstraintsMap.set(e.destination,t)),t.add(e)}deleteConstraint(e){this._constraints.delete(e),this._objectConstraintsMap.get(e.destination).delete(e)}setInitState(){const e=new Set,t=new Set;for(const n of this._constraints)this._processConstraint(n,e,t,i=>i.setInitState())}update(){const e=new Set,t=new Set;for(const n of this._constraints)this._processConstraint(n,e,t,i=>i.update())}_processConstraint(e,t,n,i){if(n.has(e))return;if(t.has(e))throw new Error("VRMNodeConstraintManager: Circular dependency detected while updating constraints");t.add(e);const r=e.dependencies;for(const o of r)_s(o,a=>{const u=this._objectConstraintsMap.get(a);if(u)for(const s of u)this._processConstraint(s,t,n,i)});i(e),n.add(e)}},vs=new S,Ms=new S,xs=class extends At{get dependencies(){return new Set([this.source])}constructor(e,t){super(e,t),this._dstRestQuat=new S,this._invSrcRestQuat=new S}setInitState(){this._dstRestQuat.copy(this.destination.quaternion),je(this._invSrcRestQuat.copy(this.source.quaternion))}update(){const e=vs.copy(this._invSrcRestQuat).multiply(this.source.quaternion),t=Ms.copy(this._dstRestQuat).multiply(e);this.destination.quaternion.copy(this._dstRestQuat).slerp(t,this.weight)}},ys=new v,ws=new S,Ts=new S,Rs=class extends At{get rollAxis(){return this._rollAxis}set rollAxis(e){this._rollAxis=e,this._v3RollAxis.set(e==="X"?1:0,e==="Y"?1:0,e==="Z"?1:0)}get dependencies(){return new Set([this.source])}constructor(e,t){super(e,t),this._rollAxis="X",this._v3RollAxis=new v(1,0,0),this._dstRestQuat=new S,this._invDstRestQuat=new S,this._invSrcRestQuatMulDstRestQuat=new S}setInitState(){this._dstRestQuat.copy(this.destination.quaternion),je(this._invDstRestQuat.copy(this._dstRestQuat)),je(this._invSrcRestQuatMulDstRestQuat.copy(this.source.quaternion)).multiply(this._dstRestQuat)}update(){const e=ws.copy(this._invDstRestQuat).multiply(this.source.quaternion).multiply(this._invSrcRestQuatMulDstRestQuat),t=ys.copy(this._v3RollAxis).applyQuaternion(e),i=Ts.setFromUnitVectors(t,this._v3RollAxis).premultiply(this._dstRestQuat).multiply(e);this.destination.quaternion.copy(this._dstRestQuat).slerp(i,this.weight)}},Ss=new Set(["1.0","1.0-beta"]),Si=class Ee{get name(){return Ee.EXTENSION_NAME}constructor(t,n){this.parser=t,this.helperRoot=n==null?void 0:n.helperRoot}afterRoot(t){return kn(this,null,function*(){t.userData.vrmNodeConstraintManager=yield this._import(t)})}_import(t){return kn(this,null,function*(){var n;const i=this.parser.json;if(!(((n=i.extensionsUsed)==null?void 0:n.indexOf(Ee.EXTENSION_NAME))!==-1))return null;const o=new gs,a=yield this.parser.getDependencies("node");return a.forEach((u,s)=>{var l;const d=i.nodes[s],c=(l=d==null?void 0:d.extensions)==null?void 0:l[Ee.EXTENSION_NAME];if(c==null)return;const p=c.specVersion;if(!Ss.has(p)){console.warn(`VRMNodeConstraintLoaderPlugin: Unknown ${Ee.EXTENSION_NAME} specVersion "${p}"`);return}const h=c.constraint;if(h.roll!=null){const _=this._importRollConstraint(u,a,h.roll);o.addConstraint(_)}else if(h.aim!=null){const _=this._importAimConstraint(u,a,h.aim);o.addConstraint(_)}else if(h.rotation!=null){const _=this._importRotationConstraint(u,a,h.rotation);o.addConstraint(_)}}),t.scene.updateMatrixWorld(),o.setInitState(),o})}_importRollConstraint(t,n,i){const{source:r,rollAxis:o,weight:a}=i,u=n[r],s=new Rs(t,u);if(o!=null&&(s.rollAxis=o),a!=null&&(s.weight=a),this.helperRoot){const l=new it(s);this.helperRoot.add(l)}return s}_importAimConstraint(t,n,i){const{source:r,aimAxis:o,weight:a}=i,u=n[r],s=new ms(t,u);if(o!=null&&(s.aimAxis=o),a!=null&&(s.weight=a),this.helperRoot){const l=new it(s);this.helperRoot.add(l)}return s}_importRotationConstraint(t,n,i){const{source:r,weight:o}=i,a=n[r],u=new xs(t,a);if(o!=null&&(u.weight=o),this.helperRoot){const s=new it(u);this.helperRoot.add(s)}return u}};Si.EXTENSION_NAME="VRMC_node_constraint";var As=Si,De=(e,t,n)=>new Promise((i,r)=>{var o=s=>{try{u(n.next(s))}catch(l){r(l)}},a=s=>{try{u(n.throw(s))}catch(l){r(l)}},u=s=>s.done?i(s.value):Promise.resolve(s.value).then(o,a);u((n=n.apply(e,t)).next())}),Et=class{},rt=new v,ae=new v,Ai=class extends Et{get type(){return"capsule"}constructor(e){var t,n,i,r;super(),this.offset=(t=e==null?void 0:e.offset)!=null?t:new v(0,0,0),this.tail=(n=e==null?void 0:e.tail)!=null?n:new v(0,0,0),this.radius=(i=e==null?void 0:e.radius)!=null?i:0,this.inside=(r=e==null?void 0:e.inside)!=null?r:!1}calculateCollision(e,t,n,i){rt.setFromMatrixPosition(e),ae.subVectors(this.tail,this.offset).applyMatrix4(e),ae.sub(rt);const r=ae.lengthSq();i.copy(t).sub(rt);const o=ae.dot(i);o<=0||(r<=o||ae.multiplyScalar(o/r),i.sub(ae));const a=i.length(),u=this.inside?this.radius-n-a:a-n-this.radius;return u<0&&(i.multiplyScalar(1/a),this.inside&&i.negate()),u}},ot=new v,Bn=new $,Ei=class extends Et{get type(){return"plane"}constructor(e){var t,n;super(),this.offset=(t=e==null?void 0:e.offset)!=null?t:new v(0,0,0),this.normal=(n=e==null?void 0:e.normal)!=null?n:new v(0,0,1)}calculateCollision(e,t,n,i){i.setFromMatrixPosition(e),i.negate().add(t),Bn.getNormalMatrix(e),ot.copy(this.normal).applyNormalMatrix(Bn).normalize();const r=i.dot(ot)-n;return i.copy(ot),r}},Es=new v,Pi=class extends Et{get type(){return"sphere"}constructor(e){var t,n,i;super(),this.offset=(t=e==null?void 0:e.offset)!=null?t:new v(0,0,0),this.radius=(n=e==null?void 0:e.radius)!=null?n:0,this.inside=(i=e==null?void 0:e.inside)!=null?i:!1}calculateCollision(e,t,n,i){i.subVectors(t,Es.setFromMatrixPosition(e));const r=i.length(),o=this.inside?this.radius-n-r:r-n-this.radius;return o<0&&(i.multiplyScalar(1/r),this.inside&&i.negate()),o}},Q=new v,Ps=class extends ee{constructor(e){super(),this.worldScale=1,this._currentRadius=0,this._currentOffset=new v,this._currentTail=new v,this._shape=e,this._attrPos=new F(new Float32Array(396),3),this.setAttribute("position",this._attrPos),this._attrIndex=new F(new Uint16Array(264),1),this.setIndex(this._attrIndex),this._buildIndex(),this.update()}update(){let e=!1;const t=this._shape.radius/this.worldScale;this._currentRadius!==t&&(this._currentRadius=t,e=!0),this._currentOffset.equals(this._shape.offset)||(this._currentOffset.copy(this._shape.offset),e=!0);const n=Q.copy(this._shape.tail).divideScalar(this.worldScale);this._currentTail.distanceToSquared(n)>1e-10&&(this._currentTail.copy(n),e=!0),e&&this._buildPosition()}_buildPosition(){Q.copy(this._currentTail).sub(this._currentOffset);const e=Q.length()/this._currentRadius;for(let i=0;i<=16;i++){const r=i/16*Math.PI;this._attrPos.setXYZ(i,-Math.sin(r),-Math.cos(r),0),this._attrPos.setXYZ(17+i,e+Math.sin(r),Math.cos(r),0),this._attrPos.setXYZ(34+i,-Math.sin(r),0,-Math.cos(r)),this._attrPos.setXYZ(51+i,e+Math.sin(r),0,Math.cos(r))}for(let i=0;i<32;i++){const r=i/16*Math.PI;this._attrPos.setXYZ(68+i,0,Math.sin(r),Math.cos(r)),this._attrPos.setXYZ(100+i,e,Math.sin(r),Math.cos(r))}const t=Math.atan2(Q.y,Math.sqrt(Q.x*Q.x+Q.z*Q.z)),n=-Math.atan2(Q.z,Q.x);this.rotateZ(t),this.rotateY(n),this.scale(this._currentRadius,this._currentRadius,this._currentRadius),this.translate(this._currentOffset.x,this._currentOffset.y,this._currentOffset.z),this._attrPos.needsUpdate=!0}_buildIndex(){for(let e=0;e<34;e++){const t=(e+1)%34;this._attrIndex.setXY(e*2,e,t),this._attrIndex.setXY(68+e*2,34+e,34+t)}for(let e=0;e<32;e++){const t=(e+1)%32;this._attrIndex.setXY(136+e*2,68+e,68+t),this._attrIndex.setXY(200+e*2,100+e,100+t)}this._attrIndex.needsUpdate=!0}},Ls=class extends ee{constructor(e){super(),this.worldScale=1,this._currentOffset=new v,this._currentNormal=new v,this._shape=e,this._attrPos=new F(new Float32Array(18),3),this.setAttribute("position",this._attrPos),this._attrIndex=new F(new Uint16Array(10),1),this.setIndex(this._attrIndex),this._buildIndex(),this.update()}update(){let e=!1;this._currentOffset.equals(this._shape.offset)||(this._currentOffset.copy(this._shape.offset),e=!0),this._currentNormal.equals(this._shape.normal)||(this._currentNormal.copy(this._shape.normal),e=!0),e&&this._buildPosition()}_buildPosition(){this._attrPos.setXYZ(0,-.5,-.5,0),this._attrPos.setXYZ(1,.5,-.5,0),this._attrPos.setXYZ(2,.5,.5,0),this._attrPos.setXYZ(3,-.5,.5,0),this._attrPos.setXYZ(4,0,0,0),this._attrPos.setXYZ(5,0,0,.25),this.translate(this._currentOffset.x,this._currentOffset.y,this._currentOffset.z),this.lookAt(this._currentNormal),this._attrPos.needsUpdate=!0}_buildIndex(){this._attrIndex.setXY(0,0,1),this._attrIndex.setXY(2,1,2),this._attrIndex.setXY(4,2,3),this._attrIndex.setXY(6,3,0),this._attrIndex.setXY(8,4,5),this._attrIndex.needsUpdate=!0}},bs=class extends ee{constructor(e){super(),this.worldScale=1,this._currentRadius=0,this._currentOffset=new v,this._shape=e,this._attrPos=new F(new Float32Array(288),3),this.setAttribute("position",this._attrPos),this._attrIndex=new F(new Uint16Array(192),1),this.setIndex(this._attrIndex),this._buildIndex(),this.update()}update(){let e=!1;const t=this._shape.radius/this.worldScale;this._currentRadius!==t&&(this._currentRadius=t,e=!0),this._currentOffset.equals(this._shape.offset)||(this._currentOffset.copy(this._shape.offset),e=!0),e&&this._buildPosition()}_buildPosition(){for(let e=0;e<32;e++){const t=e/16*Math.PI;this._attrPos.setXYZ(e,Math.cos(t),Math.sin(t),0),this._attrPos.setXYZ(32+e,0,Math.cos(t),Math.sin(t)),this._attrPos.setXYZ(64+e,Math.sin(t),0,Math.cos(t))}this.scale(this._currentRadius,this._currentRadius,this._currentRadius),this.translate(this._currentOffset.x,this._currentOffset.y,this._currentOffset.z),this._attrPos.needsUpdate=!0}_buildIndex(){for(let e=0;e<32;e++){const t=(e+1)%32;this._attrIndex.setXY(e*2,e,t),this._attrIndex.setXY(64+e*2,32+e,32+t),this._attrIndex.setXY(128+e*2,64+e,64+t)}this._attrIndex.needsUpdate=!0}},Is=new v,st=class extends ve{constructor(e){if(super(),this.matrixAutoUpdate=!1,this.collider=e,this.collider.shape instanceof Pi)this._geometry=new bs(this.collider.shape);else if(this.collider.shape instanceof Ai)this._geometry=new Ps(this.collider.shape);else if(this.collider.shape instanceof Ei)this._geometry=new Ls(this.collider.shape);else throw new Error("VRMSpringBoneColliderHelper: Unknown collider shape type detected");const t=new qe({color:16711935,depthTest:!1,depthWrite:!1});this._line=new wt(this._geometry,t),this.add(this._line)}dispose(){this._geometry.dispose()}updateMatrixWorld(e){this.collider.updateWorldMatrix(!0,!1),this.matrix.copy(this.collider.matrixWorld);const t=this.matrix.elements;this._geometry.worldScale=Is.set(t[0],t[1],t[2]).length(),this._geometry.update(),super.updateMatrixWorld(e)}},Cs=class extends ee{constructor(e){super(),this.worldScale=1,this._currentRadius=0,this._currentTail=new v,this._springBone=e,this._attrPos=new F(new Float32Array(294),3),this.setAttribute("position",this._attrPos),this._attrIndex=new F(new Uint16Array(194),1),this.setIndex(this._attrIndex),this._buildIndex(),this.update()}update(){let e=!1;const t=this._springBone.settings.hitRadius/this.worldScale;this._currentRadius!==t&&(this._currentRadius=t,e=!0),this._currentTail.equals(this._springBone.initialLocalChildPosition)||(this._currentTail.copy(this._springBone.initialLocalChildPosition),e=!0),e&&this._buildPosition()}_buildPosition(){for(let e=0;e<32;e++){const t=e/16*Math.PI;this._attrPos.setXYZ(e,Math.cos(t),Math.sin(t),0),this._attrPos.setXYZ(32+e,0,Math.cos(t),Math.sin(t)),this._attrPos.setXYZ(64+e,Math.sin(t),0,Math.cos(t))}this.scale(this._currentRadius,this._currentRadius,this._currentRadius),this.translate(this._currentTail.x,this._currentTail.y,this._currentTail.z),this._attrPos.setXYZ(96,0,0,0),this._attrPos.setXYZ(97,this._currentTail.x,this._currentTail.y,this._currentTail.z),this._attrPos.needsUpdate=!0}_buildIndex(){for(let e=0;e<32;e++){const t=(e+1)%32;this._attrIndex.setXY(e*2,e,t),this._attrIndex.setXY(64+e*2,32+e,32+t),this._attrIndex.setXY(128+e*2,64+e,64+t)}this._attrIndex.setXY(192,96,97),this._attrIndex.needsUpdate=!0}},Ns=new v,Os=class extends ve{constructor(e){super(),this.matrixAutoUpdate=!1,this.springBone=e,this._geometry=new Cs(this.springBone);const t=new qe({color:16776960,depthTest:!1,depthWrite:!1});this._line=new wt(this._geometry,t),this.add(this._line)}dispose(){this._geometry.dispose()}updateMatrixWorld(e){this.springBone.bone.updateWorldMatrix(!0,!1),this.matrix.copy(this.springBone.bone.matrixWorld);const t=this.matrix.elements;this._geometry.worldScale=Ns.set(t[0],t[1],t[2]).length(),this._geometry.update(),super.updateMatrixWorld(e)}},at=class extends ge{constructor(e){super(),this.colliderMatrix=new J,this.shape=e}updateWorldMatrix(e,t){super.updateWorldMatrix(e,t),Us(this.colliderMatrix,this.matrixWorld,this.shape.offset)}};function Us(e,t,n){const i=t.elements;e.copy(t),n&&(e.elements[12]=i[0]*n.x+i[4]*n.y+i[8]*n.z+i[12],e.elements[13]=i[1]*n.x+i[5]*n.y+i[9]*n.z+i[13],e.elements[14]=i[2]*n.x+i[6]*n.y+i[10]*n.z+i[14])}var Vs=new J;function ks(e){return e.invert?e.invert():e.getInverse(Vs.copy(e)),e}var Ds=class{constructor(e){this._inverseCache=new J,this._shouldUpdateInverse=!0,this.matrix=e;const t={set:(n,i,r)=>(this._shouldUpdateInverse=!0,n[i]=r,!0)};this._originalElements=e.elements,e.elements=new Proxy(e.elements,t)}get inverse(){return this._shouldUpdateInverse&&(ks(this._inverseCache.copy(this.matrix)),this._shouldUpdateInverse=!1),this._inverseCache}revert(){this.matrix.elements=this._originalElements}},lt=new J,he=new v,Te=new v,Re=new v,Se=new v,Bs=new J,Fs=class{constructor(e,t,n={},i=[]){this._currentTail=new v,this._prevTail=new v,this._boneAxis=new v,this._worldSpaceBoneLength=0,this._center=null,this._initialLocalMatrix=new J,this._initialLocalRotation=new S,this._initialLocalChildPosition=new v;var r,o,a,u,s,l;this.bone=e,this.bone.matrixAutoUpdate=!1,this.child=t,this.settings={hitRadius:(r=n.hitRadius)!=null?r:0,stiffness:(o=n.stiffness)!=null?o:1,gravityPower:(a=n.gravityPower)!=null?a:0,gravityDir:(s=(u=n.gravityDir)==null?void 0:u.clone())!=null?s:new v(0,-1,0),dragForce:(l=n.dragForce)!=null?l:.4},this.colliderGroups=i}get dependencies(){const e=new Set,t=this.bone.parent;t&&e.add(t);for(let n=0;n<this.colliderGroups.length;n++)for(let i=0;i<this.colliderGroups[n].colliders.length;i++)e.add(this.colliderGroups[n].colliders[i]);return e}get center(){return this._center}set center(e){var t;(t=this._center)!=null&&t.userData.inverseCacheProxy&&(this._center.userData.inverseCacheProxy.revert(),delete this._center.userData.inverseCacheProxy),this._center=e,this._center&&(this._center.userData.inverseCacheProxy||(this._center.userData.inverseCacheProxy=new Ds(this._center.matrixWorld)))}get initialLocalChildPosition(){return this._initialLocalChildPosition}get _parentMatrixWorld(){return this.bone.parent?this.bone.parent.matrixWorld:lt}setInitState(){this._initialLocalMatrix.copy(this.bone.matrix),this._initialLocalRotation.copy(this.bone.quaternion),this.child?this._initialLocalChildPosition.copy(this.child.position):this._initialLocalChildPosition.copy(this.bone.position).normalize().multiplyScalar(.07);const e=this._getMatrixWorldToCenter();this.bone.localToWorld(this._currentTail.copy(this._initialLocalChildPosition)).applyMatrix4(e),this._prevTail.copy(this._currentTail),this._boneAxis.copy(this._initialLocalChildPosition).normalize()}reset(){this.bone.quaternion.copy(this._initialLocalRotation),this.bone.updateMatrix(),this.bone.matrixWorld.multiplyMatrices(this._parentMatrixWorld,this.bone.matrix);const e=this._getMatrixWorldToCenter();this.bone.localToWorld(this._currentTail.copy(this._initialLocalChildPosition)).applyMatrix4(e),this._prevTail.copy(this._currentTail)}update(e){if(e<=0)return;this._calcWorldSpaceBoneLength();const t=Te.copy(this._boneAxis).transformDirection(this._initialLocalMatrix).transformDirection(this._parentMatrixWorld);Se.copy(this._currentTail).add(he.subVectors(this._currentTail,this._prevTail).multiplyScalar(1-this.settings.dragForce)).applyMatrix4(this._getMatrixCenterToWorld()).addScaledVector(t,this.settings.stiffness*e).addScaledVector(this.settings.gravityDir,this.settings.gravityPower*e),Re.setFromMatrixPosition(this.bone.matrixWorld),Se.sub(Re).normalize().multiplyScalar(this._worldSpaceBoneLength).add(Re),this._collision(Se),this._prevTail.copy(this._currentTail),this._currentTail.copy(Se).applyMatrix4(this._getMatrixWorldToCenter());const n=Bs.multiplyMatrices(this._parentMatrixWorld,this._initialLocalMatrix).invert();this.bone.quaternion.setFromUnitVectors(this._boneAxis,he.copy(Se).applyMatrix4(n).normalize()).premultiply(this._initialLocalRotation),this.bone.updateMatrix(),this.bone.matrixWorld.multiplyMatrices(this._parentMatrixWorld,this.bone.matrix)}_collision(e){for(let t=0;t<this.colliderGroups.length;t++)for(let n=0;n<this.colliderGroups[t].colliders.length;n++){const i=this.colliderGroups[t].colliders[n],r=i.shape.calculateCollision(i.colliderMatrix,e,this.settings.hitRadius,he);if(r<0){e.addScaledVector(he,-r),e.sub(Re);const o=e.length();e.multiplyScalar(this._worldSpaceBoneLength/o).add(Re)}}}_calcWorldSpaceBoneLength(){he.setFromMatrixPosition(this.bone.matrixWorld),this.child?Te.setFromMatrixPosition(this.child.matrixWorld):(Te.copy(this._initialLocalChildPosition),Te.applyMatrix4(this.bone.matrixWorld)),this._worldSpaceBoneLength=he.sub(Te).length()}_getMatrixCenterToWorld(){return this._center?this._center.matrixWorld:lt}_getMatrixWorldToCenter(){return this._center?this._center.userData.inverseCacheProxy.inverse:lt}};function Hs(e,t){const n=[];let i=e;for(;i!==null;)n.unshift(i),i=i.parent;n.forEach(r=>{t(r)})}function xt(e,t){e.children.forEach(n=>{t(n)||xt(n,t)})}function Ws(e){var t;const n=new Map;for(const i of e){let r=i;do{const o=((t=n.get(r))!=null?t:0)+1;if(o===e.size)return r;n.set(r,o),r=r.parent}while(r!==null)}return null}var Fn=class{constructor(){this._joints=new Set,this._sortedJoints=[],this._hasWarnedCircularDependency=!1,this._ancestors=[],this._objectSpringBonesMap=new Map,this._isSortedJointsDirty=!1,this._relevantChildrenUpdated=this._relevantChildrenUpdated.bind(this)}get joints(){return this._joints}get springBones(){return console.warn("VRMSpringBoneManager: springBones is deprecated. use joints instead."),this._joints}get colliderGroups(){const e=new Set;return this._joints.forEach(t=>{t.colliderGroups.forEach(n=>{e.add(n)})}),Array.from(e)}get colliders(){const e=new Set;return this.colliderGroups.forEach(t=>{t.colliders.forEach(n=>{e.add(n)})}),Array.from(e)}addJoint(e){this._joints.add(e);let t=this._objectSpringBonesMap.get(e.bone);t==null&&(t=new Set,this._objectSpringBonesMap.set(e.bone,t)),t.add(e),this._isSortedJointsDirty=!0}addSpringBone(e){console.warn("VRMSpringBoneManager: addSpringBone() is deprecated. use addJoint() instead."),this.addJoint(e)}deleteJoint(e){this._joints.delete(e),this._objectSpringBonesMap.get(e.bone).delete(e),this._isSortedJointsDirty=!0}deleteSpringBone(e){console.warn("VRMSpringBoneManager: deleteSpringBone() is deprecated. use deleteJoint() instead."),this.deleteJoint(e)}setInitState(){this._sortJoints();for(let e=0;e<this._sortedJoints.length;e++){const t=this._sortedJoints[e];t.bone.updateMatrix(),t.bone.updateWorldMatrix(!1,!1),t.setInitState()}}reset(){this._sortJoints();for(let e=0;e<this._sortedJoints.length;e++){const t=this._sortedJoints[e];t.bone.updateMatrix(),t.bone.updateWorldMatrix(!1,!1),t.reset()}}update(e){this._sortJoints();for(let t=0;t<this._ancestors.length;t++)this._ancestors[t].updateWorldMatrix(t===0,!1);for(let t=0;t<this._sortedJoints.length;t++){const n=this._sortedJoints[t];n.bone.updateMatrix(),n.bone.updateWorldMatrix(!1,!1),n.update(e),xt(n.bone,this._relevantChildrenUpdated)}}_sortJoints(){if(!this._isSortedJointsDirty)return;const e=[],t=new Set,n=new Set,i=new Set;for(const o of this._joints)this._insertJointSort(o,t,n,e,i);this._sortedJoints=e;const r=Ws(i);this._ancestors=[],r&&(this._ancestors.push(r),xt(r,o=>{var a,u;return((u=(a=this._objectSpringBonesMap.get(o))==null?void 0:a.size)!=null?u:0)>0?!0:(this._ancestors.push(o),!1)})),this._isSortedJointsDirty=!1}_insertJointSort(e,t,n,i,r){if(n.has(e))return;if(t.has(e)){this._hasWarnedCircularDependency||(console.warn("VRMSpringBoneManager: Circular dependency detected"),this._hasWarnedCircularDependency=!0);return}t.add(e);const o=e.dependencies;for(const a of o){let u=!1,s=null;Hs(a,l=>{const d=this._objectSpringBonesMap.get(l);if(d)for(const c of d)u=!0,this._insertJointSort(c,t,n,i,r);else u||(s=l)}),s&&r.add(s)}i.push(e),n.add(e)}_relevantChildrenUpdated(e){var t,n;return((n=(t=this._objectSpringBonesMap.get(e))==null?void 0:t.size)!=null?n:0)>0?!0:(e.updateWorldMatrix(!1,!1),!1)}},Hn="VRMC_springBone_extended_collider",zs=new Set(["1.0","1.0-beta"]),js=new Set(["1.0"]),Li=class fe{get name(){return fe.EXTENSION_NAME}constructor(t,n){var i;this.parser=t,this.jointHelperRoot=n==null?void 0:n.jointHelperRoot,this.colliderHelperRoot=n==null?void 0:n.colliderHelperRoot,this.useExtendedColliders=(i=n==null?void 0:n.useExtendedColliders)!=null?i:!0}afterRoot(t){return De(this,null,function*(){t.userData.vrmSpringBoneManager=yield this._import(t)})}_import(t){return De(this,null,function*(){const n=yield this._v1Import(t);if(n!=null)return n;const i=yield this._v0Import(t);return i??null})}_v1Import(t){return De(this,null,function*(){var n,i,r,o,a;const u=t.parser.json;if(!(((n=u.extensionsUsed)==null?void 0:n.indexOf(fe.EXTENSION_NAME))!==-1))return null;const l=new Fn,d=yield t.parser.getDependencies("node"),c=(i=u.extensions)==null?void 0:i[fe.EXTENSION_NAME];if(!c)return null;const p=c.specVersion;if(!zs.has(p))return console.warn(`VRMSpringBoneLoaderPlugin: Unknown ${fe.EXTENSION_NAME} specVersion "${p}"`),null;const h=(r=c.colliders)==null?void 0:r.map((g,m)=>{var f,x,T,y,w,R,A,P,N,b,C,D,W,G,M;const L=d[g.node];if(L==null)return console.warn(`VRMSpringBoneLoaderPlugin: The collider #${m} attempted to reference a node #${g.node} but not found. Skipping the collider`),null;const B=g.shape,z=(f=g.extensions)==null?void 0:f[Hn];if(this.useExtendedColliders&&z!=null){const X=z.specVersion;if(!js.has(X))console.warn(`VRMSpringBoneLoaderPlugin: Unknown ${Hn} specVersion "${X}". Fallbacking to the ${fe.EXTENSION_NAME} definition`);else{const V=z.shape;if(V.sphere)return this._importSphereCollider(L,{offset:new v().fromArray((x=V.sphere.offset)!=null?x:[0,0,0]),radius:(T=V.sphere.radius)!=null?T:0,inside:(y=V.sphere.inside)!=null?y:!1});if(V.capsule)return this._importCapsuleCollider(L,{offset:new v().fromArray((w=V.capsule.offset)!=null?w:[0,0,0]),radius:(R=V.capsule.radius)!=null?R:0,tail:new v().fromArray((A=V.capsule.tail)!=null?A:[0,0,0]),inside:(P=V.capsule.inside)!=null?P:!1});if(V.plane)return this._importPlaneCollider(L,{offset:new v().fromArray((N=V.plane.offset)!=null?N:[0,0,0]),normal:new v().fromArray((b=V.plane.normal)!=null?b:[0,0,1])})}}if(B.sphere)return this._importSphereCollider(L,{offset:new v().fromArray((C=B.sphere.offset)!=null?C:[0,0,0]),radius:(D=B.sphere.radius)!=null?D:0,inside:!1});if(B.capsule)return this._importCapsuleCollider(L,{offset:new v().fromArray((W=B.capsule.offset)!=null?W:[0,0,0]),radius:(G=B.capsule.radius)!=null?G:0,tail:new v().fromArray((M=B.capsule.tail)!=null?M:[0,0,0]),inside:!1});console.warn(`VRMSpringBoneLoaderPlugin: The collider #${m} has no valid shape. Skipping the collider`)}),_=(o=c.colliderGroups)==null?void 0:o.map((g,m)=>{var f;return{colliders:((f=g.colliders)!=null?f:[]).map(T=>{const y=h==null?void 0:h[T];return y??(console.warn(`VRMSpringBoneLoaderPlugin: The collider group #${m} attempted to reference a collider #${T} but not found. Skipping the collider`),null)}).filter(T=>T!=null),name:g.name}});return(a=c.springs)==null||a.forEach((g,m)=>{var f;const x=g.joints,T=(f=g.colliderGroups)==null?void 0:f.map(R=>{const A=_==null?void 0:_[R];return A??(console.warn(`VRMSpringBoneLoaderPlugin: The spring #${m} attempted to reference a collider group #${R} but not found. Skipping the collider group`),null)}).filter(R=>R!=null),y=g.center!=null?d[g.center]:void 0;let w;x.forEach(R=>{if(w){const A=w.node,P=d[A],N=R.node,b=d[N],C={hitRadius:w.hitRadius,dragForce:w.dragForce,gravityPower:w.gravityPower,stiffness:w.stiffness,gravityDir:w.gravityDir!=null?new v().fromArray(w.gravityDir):void 0},D=this._importJoint(P,b,C,T);y&&(D.center=y),l.addJoint(D)}w=R})}),l.setInitState(),l})}_v0Import(t){return De(this,null,function*(){var n,i,r;const o=t.parser.json;if(!(((n=o.extensionsUsed)==null?void 0:n.indexOf("VRM"))!==-1))return null;const u=(i=o.extensions)==null?void 0:i.VRM,s=u==null?void 0:u.secondaryAnimation;if(!s)return null;const l=s==null?void 0:s.boneGroups;if(!l)return null;const d=new Fn,c=yield t.parser.getDependencies("node"),p=(r=s.colliderGroups)==null?void 0:r.map((h,_)=>{var g;const m=c[h.node];return m==null?(console.warn(`VRMSpringBoneLoaderPlugin: The collider group #${_} attempted to reference a node #${h.node} but not found. Skipping the collider group`),null):{colliders:((g=h.colliders)!=null?g:[]).map((x,T)=>{var y,w,R;const A=new v(0,0,0);return x.offset&&A.set((y=x.offset.x)!=null?y:0,(w=x.offset.y)!=null?w:0,x.offset.z?-x.offset.z:0),this._importSphereCollider(m,{offset:A,radius:(R=x.radius)!=null?R:0,inside:!1})})}});return l==null||l.forEach((h,_)=>{const g=h.bones;g&&g.forEach(m=>{var f,x,T,y;const w=c[m];if(w==null){console.warn(`VRMSpringBoneLoaderPlugin: The spring bone group #${_} attempted to reference a node #${m} but not found. Skipping the node`);return}const R=new v;h.gravityDir?R.set((f=h.gravityDir.x)!=null?f:0,(x=h.gravityDir.y)!=null?x:0,(T=h.gravityDir.z)!=null?T:0):R.set(0,-1,0);const A=h.center!=null?c[h.center]:void 0,P={hitRadius:h.hitRadius,dragForce:h.dragForce,gravityPower:h.gravityPower,stiffness:h.stiffiness,gravityDir:R},N=(y=h.colliderGroups)==null?void 0:y.map(b=>{const C=p==null?void 0:p[b];return C??(console.warn(`VRMSpringBoneLoaderPlugin: The spring #${_} attempted to reference a collider group #${b} but not found. Skipping the collider group`),null)}).filter(b=>b!=null);w.traverse(b=>{var C;const D=(C=b.children[0])!=null?C:null,W=this._importJoint(b,D,P,N);A&&(W.center=A),d.addJoint(W)})})}),t.scene.updateMatrixWorld(),d.setInitState(),d})}_importJoint(t,n,i,r){const o=new Fs(t,n,i,r);if(this.jointHelperRoot){const a=new Os(o);this.jointHelperRoot.add(a),a.renderOrder=this.jointHelperRoot.renderOrder}return o}_importSphereCollider(t,n){const i=new Pi(n),r=new at(i);if(t.add(r),this.colliderHelperRoot){const o=new st(r);this.colliderHelperRoot.add(o),o.renderOrder=this.colliderHelperRoot.renderOrder}return r}_importCapsuleCollider(t,n){const i=new Ai(n),r=new at(i);if(t.add(r),this.colliderHelperRoot){const o=new st(r);this.colliderHelperRoot.add(o),o.renderOrder=this.colliderHelperRoot.renderOrder}return r}_importPlaneCollider(t,n){const i=new Ei(n),r=new at(i);if(t.add(r),this.colliderHelperRoot){const o=new st(r);this.colliderHelperRoot.add(o),o.renderOrder=this.colliderHelperRoot.renderOrder}return r}};Li.EXTENSION_NAME="VRMC_springBone";var Gs=Li,Xs=class{get name(){return"VRMLoaderPlugin"}constructor(e,t){var n,i,r,o,a,u,s,l,d,c;this.parser=e;const p=t==null?void 0:t.helperRoot,h=t==null?void 0:t.autoUpdateHumanBones;this.expressionPlugin=(n=t==null?void 0:t.expressionPlugin)!=null?n:new ao(e),this.firstPersonPlugin=(i=t==null?void 0:t.firstPersonPlugin)!=null?i:new uo(e),this.humanoidPlugin=(r=t==null?void 0:t.humanoidPlugin)!=null?r:new go(e,{helperRoot:p,autoUpdateHumanBones:h}),this.lookAtPlugin=(o=t==null?void 0:t.lookAtPlugin)!=null?o:new Co(e,{helperRoot:p}),this.metaPlugin=(a=t==null?void 0:t.metaPlugin)!=null?a:new Uo(e),this.mtoonMaterialPlugin=(u=t==null?void 0:t.mtoonMaterialPlugin)!=null?u:new $o(e),this.materialsHDREmissiveMultiplierPlugin=(s=t==null?void 0:t.materialsHDREmissiveMultiplierPlugin)!=null?s:new Jo(e),this.materialsV0CompatPlugin=(l=t==null?void 0:t.materialsV0CompatPlugin)!=null?l:new os(e),this.springBonePlugin=(d=t==null?void 0:t.springBonePlugin)!=null?d:new Gs(e,{colliderHelperRoot:p,jointHelperRoot:p}),this.nodeConstraintPlugin=(c=t==null?void 0:t.nodeConstraintPlugin)!=null?c:new As(e,{helperRoot:p})}beforeRoot(){return Ue(this,null,function*(){yield this.materialsV0CompatPlugin.beforeRoot(),yield this.mtoonMaterialPlugin.beforeRoot()})}loadMesh(e){return Ue(this,null,function*(){return yield this.mtoonMaterialPlugin.loadMesh(e)})}getMaterialType(e){const t=this.mtoonMaterialPlugin.getMaterialType(e);return t??null}extendMaterialParams(e,t){return Ue(this,null,function*(){yield this.materialsHDREmissiveMultiplierPlugin.extendMaterialParams(e,t),yield this.mtoonMaterialPlugin.extendMaterialParams(e,t)})}afterRoot(e){return Ue(this,null,function*(){yield this.metaPlugin.afterRoot(e),yield this.humanoidPlugin.afterRoot(e),yield this.expressionPlugin.afterRoot(e),yield this.lookAtPlugin.afterRoot(e),yield this.firstPersonPlugin.afterRoot(e),yield this.springBonePlugin.afterRoot(e),yield this.nodeConstraintPlugin.afterRoot(e),yield this.mtoonMaterialPlugin.afterRoot(e);const t=e.userData.vrmMeta,n=e.userData.vrmHumanoid;if(t&&n){const i=new ko({scene:e.scene,expressionManager:e.userData.vrmExpressionManager,firstPerson:e.userData.vrmFirstPerson,humanoid:n,lookAt:e.userData.vrmLookAt,meta:t,materials:e.userData.vrmMToonMaterials,springBoneManager:e.userData.vrmSpringBoneManager,nodeConstraintManager:e.userData.vrmNodeConstraintManager});e.userData.vrm=i}})}};function qs(e){const t=new Set;return e.traverse(n=>{if(!n.isMesh)return;const i=n;t.add(i)}),t}function Wn(e,t,n){if(t.size===1){const a=t.values().next().value;if(a.weight===1)return e[a.index]}const i=new Float32Array(e[0].count*3);let r=0;if(n)r=1;else for(const a of t)r+=a.weight;for(const a of t){const u=e[a.index],s=a.weight/r;for(let l=0;l<u.count;l++)i[l*3+0]+=u.getX(l)*s,i[l*3+1]+=u.getY(l)*s,i[l*3+2]+=u.getZ(l)*s}return new F(i,3)}function Ys(e){var t;const n=qs(e.scene),i=new Map,r=(t=e.expressionManager)==null?void 0:t.expressionMap;if(r!=null)for(const[o,a]of Object.entries(r)){const u=new Set;for(const s of a.binds)if(s instanceof ze){if(s.weight!==0)for(const l of s.primitives){let d=i.get(l);d==null&&(d=new Map,i.set(l,d));let c=d.get(o);c==null&&(c=new Set,d.set(o,c)),c.add(s)}u.add(s)}for(const s of u)a.deleteBind(s)}for(const o of n){const a=i.get(o);if(a==null)continue;const u=o.geometry.morphAttributes;o.geometry.morphAttributes={};const s=o.geometry.clone();o.geometry=s;const l=s.morphTargetsRelative,d=u.position!=null,c=u.normal!=null,p={},h={},_=[];if(d||c){d&&(p.position=[]),c&&(p.normal=[]);let g=0;for(const[m,f]of a)d&&(p.position[g]=Wn(u.position,f,l)),c&&(p.normal[g]=Wn(u.normal,f,l)),r==null||r[m].addBind(new ze({index:g,weight:1,primitives:[o]})),h[m]=g,_.push(0),g++}s.morphAttributes=p,o.morphTargetDictionary=h,o.morphTargetInfluences=_}}function Ge(e,t,n){if(e.getComponent)return e.getComponent(t,n);{let i=e.array[t*e.itemSize+n];return e.normalized&&(i=O.denormalize(i,e.array)),i}}function bi(e,t,n,i){e.setComponent?e.setComponent(t,n,i):(e.normalized&&(i=O.normalize(i,e.array)),e.array[t*e.itemSize+n]=i)}function Qs(e){var t;const n=$s(e),i=new Set;for(const c of n)i.has(c.geometry)&&(c.geometry=na(c.geometry)),i.add(c.geometry);const r=new Map;for(const c of i){const p=c.getAttribute("skinIndex"),h=(t=r.get(p))!=null?t:new Map;r.set(p,h);const _=c.getAttribute("skinWeight"),g=Zs(p,_);h.set(_,g)}const o=new Map;for(const c of n){const p=Js(c,r);o.set(c,p)}const a=[];for(const[c,p]of o){let h=!1;for(const _ of a)if(Ks(p,_.boneInverseMap)){h=!0,_.meshes.add(c);for(const[m,f]of p)_.boneInverseMap.set(m,f);break}h||a.push({boneInverseMap:p,meshes:new Set([c])})}const u=new Map,s=new ut,l=new ut,d=new ut;for(const c of a){const{boneInverseMap:p,meshes:h}=c,_=Array.from(p.keys()),g=Array.from(p.values()),m=new Tt(_,g),f=l.getOrCreate(m);for(const x of h){const T=x.geometry.getAttribute("skinIndex"),y=s.getOrCreate(T),w=x.skeleton.bones,R=w.map(N=>d.getOrCreate(N)).join(","),A=`${y};${f};${R}`;let P=u.get(A);P==null&&(P=T.clone(),ea(P,w,_),u.set(A,P)),x.geometry.setAttribute("skinIndex",P)}for(const x of h)x.bind(m,new J)}}function $s(e){const t=new Set;return e.traverse(n=>{if(!n.isSkinnedMesh)return;const i=n;t.add(i)}),t}function Zs(e,t){const n=new Set;for(let i=0;i<e.count;i++)for(let r=0;r<e.itemSize;r++){const o=Ge(e,i,r);Ge(t,i,r)!==0&&n.add(o)}return n}function Js(e,t){const n=new Map,i=e.skeleton,r=e.geometry,o=r.getAttribute("skinIndex"),a=r.getAttribute("skinWeight"),u=t.get(o),s=u==null?void 0:u.get(a);if(!s)throw new Error("Unreachable. attributeUsedIndexSetMap does not know the skin index attribute or the skin weight attribute.");for(const l of s)n.set(i.bones[l],i.boneInverses[l]);return n}function Ks(e,t){for(const[n,i]of e.entries()){const r=t.get(n);if(r!=null&&!ta(i,r))return!1}return!0}function ea(e,t,n){const i=new Map;for(const o of t)i.set(o,i.size);const r=new Map;for(const[o,a]of n.entries()){const u=i.get(a);r.set(u,o)}for(let o=0;o<e.count;o++)for(let a=0;a<e.itemSize;a++){const u=Ge(e,o,a),s=r.get(u);bi(e,o,a,s)}e.needsUpdate=!0}function ta(e,t,n){if(n=n||1e-4,e.elements.length!=t.elements.length)return!1;for(let i=0,r=e.elements.length;i<r;i++)if(Math.abs(e.elements[i]-t.elements[i])>n)return!1;return!0}var ut=class{constructor(){this._objectIndexMap=new Map,this._index=0}get(e){return this._objectIndexMap.get(e)}getOrCreate(e){let t=this._objectIndexMap.get(e);return t==null&&(t=this._index,this._objectIndexMap.set(e,t),this._index++),t}};function na(e){var t,n,i,r;const o=new ee;o.name=e.name,o.setIndex(e.index);for(const[a,u]of Object.entries(e.attributes))o.setAttribute(a,u);for(const[a,u]of Object.entries(e.morphAttributes)){const s=a;o.morphAttributes[s]=u.concat()}o.morphTargetsRelative=e.morphTargetsRelative,o.groups=[];for(const a of e.groups)o.addGroup(a.start,a.count,a.materialIndex);return o.boundingSphere=(n=(t=e.boundingSphere)==null?void 0:t.clone())!=null?n:null,o.boundingBox=(r=(i=e.boundingBox)==null?void 0:i.clone())!=null?r:null,o.drawRange.start=e.drawRange.start,o.drawRange.count=e.drawRange.count,o.userData=e.userData,o}function zn(e){if(Object.values(e).forEach(t=>{t!=null&&t.isTexture&&t.dispose()}),e.isShaderMaterial){const t=e.uniforms;t&&Object.values(t).forEach(n=>{const i=n.value;i!=null&&i.isTexture&&i.dispose()})}e.dispose()}function ia(e){const t=e.geometry;t&&t.dispose();const n=e.skeleton;n&&n.dispose();const i=e.material;i&&(Array.isArray(i)?i.forEach(r=>zn(r)):i&&zn(i))}function ra(e){e.traverse(ia)}function oa(e,t){var n,i;console.warn("VRMUtils.removeUnnecessaryJoints: removeUnnecessaryJoints is deprecated. Use combineSkeletons instead. combineSkeletons contributes more to the performance improvement. This function will be removed in the next major version.");const r=(n=t==null?void 0:t.experimentalSameBoneCounts)!=null?n:!1,o=[];e.traverse(s=>{s.type==="SkinnedMesh"&&o.push(s)});const a=new Map;let u=0;for(const s of o){const d=s.geometry.getAttribute("skinIndex");if(a.has(d))continue;const c=new Map,p=new Map;for(let h=0;h<d.count;h++)for(let _=0;_<d.itemSize;_++){const g=Ge(d,h,_);let m=c.get(g);m==null&&(m=c.size,c.set(g,m),p.set(m,g)),bi(d,h,_,m)}d.needsUpdate=!0,a.set(d,p),u=Math.max(u,c.size)}for(const s of o){const d=s.geometry.getAttribute("skinIndex"),c=a.get(d),p=[],h=[],_=r?u:c.size;for(let m=0;m<_;m++){const f=(i=c.get(m))!=null?i:0;p.push(s.skeleton.bones[f]),h.push(s.skeleton.boneInverses[f])}const g=new Tt(p,h);s.bind(g,new J)}}function sa(e,t){const n=e.position.count,i=new Array(n);let r=0;const o=t.array;for(let a=0;a<o.length;a++){const u=o[a];i[u]||(i[u]=!0,r++)}return{isVertexUsed:i,vertexCount:n,verticesUsed:r}}function aa(e){const t=[],n=[];let i=0;for(let r=0;r<e.length;r++)if(e[r]){const o=i++;t[r]=o,n[o]=r}return{originalIndexNewIndexMap:t,newIndexOriginalIndexMap:n}}function la(e,t){var n,i,r,o;t.name=e.name,t.morphTargetsRelative=e.morphTargetsRelative,e.groups.forEach(a=>{t.addGroup(a.start,a.count,a.materialIndex)}),t.boundingBox=(i=(n=e.boundingBox)==null?void 0:n.clone())!=null?i:null,t.boundingSphere=(o=(r=e.boundingSphere)==null?void 0:r.clone())!=null?o:null,t.setDrawRange(e.drawRange.start,e.drawRange.count),t.userData=e.userData}function ua(e,t,n){const i=t.array,r=new i.constructor(i.length);for(let o=0;o<i.length;o++){const a=i[o];r[o]=n[a]}e.setIndex(new F(r,t.itemSize,t.normalized))}function Xe(e,t,n){const i=e.constructor,r=new i(t.length*n);let o=!0;for(let a=0;a<t.length;a++){const s=t[a]*n,l=a*n;for(let d=0;d<n;d++){const c=e[s+d];r[l+d]=c,o=o&&c===0}}return[r,o]}function da(e){var t;const n=new Map,i=[];for(const[r,o]of Object.entries(e))if(o.isInterleavedBufferAttribute){const a=o,u=a.data,s=(t=n.get(u))!=null?t:[];n.set(u,s),s.push([r,a])}else{const a=o;i.push([r,a])}return[n,i]}function ca(e,t,n){const[i,r]=da(t);for(const[o,a]of i){const u=o.array,{stride:s}=o,[l]=Xe(u,n,s),d=new oi(l,s);d.setUsage(o.usage);for(const[c,p]of a){const{itemSize:h,offset:_,normalized:g}=p,m=new si(d,h,_,g);e.setAttribute(c,m)}}for(const[o,a]of r){const u=a.array,{itemSize:s,normalized:l}=a,[d]=Xe(u,n,s);e.setAttribute(o,new F(d,s,l))}}function ha(e){var t;const n=new Map,i=[];for(const[r,o]of Object.entries(e)){const a=r;for(let u=0;u<o.length;u++){const s=o[u];if(s.isInterleavedBufferAttribute){const l=s,d=l.data,c=(t=n.get(d))!=null?t:[];n.set(d,c),c.push([a,u,l])}else{const l=s;i.push([a,u,l])}}}return[n,i]}function pa(e,t,n){var i,r;let o=!0;const[a,u]=ha(t),s={};for(const[l,d]of a){const c=l.array,{stride:p}=l,[h,_]=Xe(c,n,p);o=o&&_;const g=new oi(h,p);g.setUsage(l.usage);for(const[m,f,x]of d){const{itemSize:T,offset:y,normalized:w}=x,R=new si(g,T,y,w);(i=s[m])!=null||(s[m]=[]),s[m][f]=R}}for(const[l,d,c]of u){const p=c,h=p.array,{itemSize:_,normalized:g}=p,[m,f]=Xe(h,n,_);o=o&&f,(r=s[l])!=null||(s[l]=[]),s[l][d]=new F(m,_,g)}e.morphAttributes=o?{}:s}function fa(e){const t=new Map;e.traverse(n=>{if(!n.isMesh)return;const i=n,r=i.geometry,o=r.index;if(o==null)return;const a=t.get(r);if(a!=null){i.geometry=a;return}const{isVertexUsed:u,vertexCount:s,verticesUsed:l}=sa(r.attributes,o);if(l===s)return;const{originalIndexNewIndexMap:d,newIndexOriginalIndexMap:c}=aa(u),p=new ee;la(r,p),t.set(r,p),ua(p,o,d),ca(p,r.attributes,c),pa(p,r.morphAttributes,c),i.geometry=p}),Array.from(t.keys()).forEach(n=>{n.dispose()})}function ma(e){var t;((t=e.meta)==null?void 0:t.metaVersion)==="0"&&(e.scene.rotation.y=Math.PI)}var Z=class{constructor(){}};Z.combineMorphs=Ys;Z.combineSkeletons=Qs;Z.deepDispose=ra;Z.removeUnnecessaryJoints=oa;Z.removeUnnecessaryVertices=fa;Z.rotateVRM0=ma;/*!
 * @pixiv/three-vrm-core v3.5.5
 * The implementation of core features of VRM, for @pixiv/three-vrm
 *
 * Copyright (c) 2019-2026 pixiv Inc.
 * @pixiv/three-vrm-core is distributed under MIT License
 * https://github.com/pixiv/three-vrm/blob/release/LICENSE
 *//*!
 * @pixiv/three-vrm-materials-mtoon v3.5.5
 * MToon (toon material) module for @pixiv/three-vrm
 *
 * Copyright (c) 2019-2026 pixiv Inc.
 * @pixiv/three-vrm-materials-mtoon is distributed under MIT License
 * https://github.com/pixiv/three-vrm/blob/release/LICENSE
 *//*!
 * @pixiv/three-vrm-materials-hdr-emissive-multiplier v3.5.5
 * Support VRMC_hdr_emissiveMultiplier for @pixiv/three-vrm
 *
 * Copyright (c) 2019-2026 pixiv Inc.
 * @pixiv/three-vrm-materials-hdr-emissive-multiplier is distributed under MIT License
 * https://github.com/pixiv/three-vrm/blob/release/LICENSE
 *//*!
 * @pixiv/three-vrm-materials-v0compat v3.5.5
 * VRM0.0 materials compatibility layer plugin for @pixiv/three-vrm
 *
 * Copyright (c) 2019-2026 pixiv Inc.
 * @pixiv/three-vrm-materials-v0compat is distributed under MIT License
 * https://github.com/pixiv/three-vrm/blob/release/LICENSE
 *//*!
 * @pixiv/three-vrm-node-constraint v3.5.5
 * Node constraint module for @pixiv/three-vrm
 *
 * Copyright (c) 2019-2026 pixiv Inc.
 * @pixiv/three-vrm-node-constraint is distributed under MIT License
 * https://github.com/pixiv/three-vrm/blob/release/LICENSE
 *//*!
 * @pixiv/three-vrm-springbone v3.5.5
 * Spring bone module for @pixiv/three-vrm
 *
 * Copyright (c) 2019-2026 pixiv Inc.
 * @pixiv/three-vrm-springbone is distributed under MIT License
 * https://github.com/pixiv/three-vrm/blob/release/LICENSE
 *//*!
 * @pixiv/three-vrm-animation v3.5.5
 * The implementation of VRM Animation
 *
 * Copyright (c) 2019-2026 pixiv Inc.
 * @pixiv/three-vrm-animation is distributed under MIT License
 * https://github.com/pixiv/three-vrm/blob/release/LICENSE
 */var jn=(e,t,n)=>new Promise((i,r)=>{var o=s=>{try{u(n.next(s))}catch(l){r(l)}},a=s=>{try{u(n.throw(s))}catch(l){r(l)}},u=s=>s.done?i(s.value):Promise.resolve(s.value).then(o,a);u((n=n.apply(e,t)).next())}),_a={Aa:"aa",Ih:"ih",Ou:"ou",Ee:"ee",Oh:"oh",Blink:"blink",Happy:"happy",Angry:"angry",Sad:"sad",Relaxed:"relaxed",LookUp:"lookUp",Surprised:"surprised",LookDown:"lookDown",LookLeft:"lookLeft",LookRight:"lookRight",BlinkLeft:"blinkLeft",BlinkRight:"blinkRight",Neutral:"neutral"};new H;new me;new v;new v;var Gn={hips:null,spine:"hips",chest:"spine",upperChest:"chest",neck:"upperChest",head:"neck",leftEye:"head",rightEye:"head",jaw:"head",leftUpperLeg:"hips",leftLowerLeg:"leftUpperLeg",leftFoot:"leftLowerLeg",leftToes:"leftFoot",rightUpperLeg:"hips",rightLowerLeg:"rightUpperLeg",rightFoot:"rightLowerLeg",rightToes:"rightFoot",leftShoulder:"upperChest",leftUpperArm:"leftShoulder",leftLowerArm:"leftUpperArm",leftHand:"leftLowerArm",rightShoulder:"upperChest",rightUpperArm:"rightShoulder",rightLowerArm:"rightUpperArm",rightHand:"rightLowerArm",leftThumbMetacarpal:"leftHand",leftThumbProximal:"leftThumbMetacarpal",leftThumbDistal:"leftThumbProximal",leftIndexProximal:"leftHand",leftIndexIntermediate:"leftIndexProximal",leftIndexDistal:"leftIndexIntermediate",leftMiddleProximal:"leftHand",leftMiddleIntermediate:"leftMiddleProximal",leftMiddleDistal:"leftMiddleIntermediate",leftRingProximal:"leftHand",leftRingIntermediate:"leftRingProximal",leftRingDistal:"leftRingIntermediate",leftLittleProximal:"leftHand",leftLittleIntermediate:"leftLittleProximal",leftLittleDistal:"leftLittleIntermediate",rightThumbMetacarpal:"rightHand",rightThumbProximal:"rightThumbMetacarpal",rightThumbDistal:"rightThumbProximal",rightIndexProximal:"rightHand",rightIndexIntermediate:"rightIndexProximal",rightIndexDistal:"rightIndexIntermediate",rightMiddleProximal:"rightHand",rightMiddleIntermediate:"rightMiddleProximal",rightMiddleDistal:"rightMiddleIntermediate",rightRingProximal:"rightHand",rightRingIntermediate:"rightRingProximal",rightRingDistal:"rightRingIntermediate",rightLittleProximal:"rightHand",rightLittleIntermediate:"rightLittleProximal",rightLittleDistal:"rightLittleIntermediate"};function ga(e){return e.invert?e.invert():e.inverse(),e}new v;new v;new v;new v;new v;new v(0,1,0);var va=new v,Ma=new v;function xa(e,t){return e.matrixWorld.decompose(va,t,Ma),t}function dt(e){return[Math.atan2(-e.z,e.x),Math.atan2(e.y,Math.sqrt(e.x*e.x+e.z*e.z))]}function Xn(e){const t=Math.round(e/2/Math.PI);return e-2*Math.PI*t}var qn=new v(0,0,1),ya=new v,wa=new v,Ta=new v,Ra=new S,ct=new S,Yn=new S,Sa=new S,ht=new ie,Ii=class Ci{constructor(t,n){this.offsetFromHeadBone=new v,this.autoUpdate=!0,this.faceFront=new v(0,0,1),this.humanoid=t,this.applier=n,this._yaw=0,this._pitch=0,this._needsUpdate=!0,this._restHeadWorldQuaternion=this.getLookAtWorldQuaternion(new S)}get yaw(){return this._yaw}set yaw(t){this._yaw=t,this._needsUpdate=!0}get pitch(){return this._pitch}set pitch(t){this._pitch=t,this._needsUpdate=!0}get euler(){return console.warn("VRMLookAt: euler is deprecated. use getEuler() instead."),this.getEuler(new ie)}getEuler(t){return t.set(O.DEG2RAD*this._pitch,O.DEG2RAD*this._yaw,0,"YXZ")}copy(t){if(this.humanoid!==t.humanoid)throw new Error("VRMLookAt: humanoid must be same in order to copy");return this.offsetFromHeadBone.copy(t.offsetFromHeadBone),this.applier=t.applier,this.autoUpdate=t.autoUpdate,this.target=t.target,this.faceFront.copy(t.faceFront),this}clone(){return new Ci(this.humanoid,this.applier).copy(this)}reset(){this._yaw=0,this._pitch=0,this._needsUpdate=!0}getLookAtWorldPosition(t){const n=this.humanoid.getRawBoneNode("head");return t.copy(this.offsetFromHeadBone).applyMatrix4(n.matrixWorld)}getLookAtWorldQuaternion(t){const n=this.humanoid.getRawBoneNode("head");return xa(n,t)}getFaceFrontQuaternion(t){if(this.faceFront.distanceToSquared(qn)<.01)return t.copy(this._restHeadWorldQuaternion).invert();const[n,i]=dt(this.faceFront);return ht.set(0,.5*Math.PI+n,i,"YZX"),t.setFromEuler(ht).premultiply(Sa.copy(this._restHeadWorldQuaternion).invert())}getLookAtWorldDirection(t){return this.getLookAtWorldQuaternion(ct),this.getFaceFrontQuaternion(Yn),t.copy(qn).applyQuaternion(ct).applyQuaternion(Yn).applyEuler(this.getEuler(ht))}lookAt(t){const n=Ra.copy(this._restHeadWorldQuaternion).multiply(ga(this.getLookAtWorldQuaternion(ct))),i=this.getLookAtWorldPosition(wa),r=Ta.copy(t).sub(i).applyQuaternion(n).normalize(),[o,a]=dt(this.faceFront),[u,s]=dt(r),l=Xn(u-o),d=Xn(a-s);this._yaw=O.RAD2DEG*l,this._pitch=O.RAD2DEG*d,this._needsUpdate=!0}update(t){this.target!=null&&this.autoUpdate&&this.lookAt(this.target.getWorldPosition(ya)),this._needsUpdate&&(this._needsUpdate=!1,this.applier.applyYawPitch(this._yaw,this._pitch))}};Ii.EULER_ORDER="YXZ";var Aa=Ii;new v(0,0,1);new ie(0,0,0,"YXZ");var Qn=180/Math.PI,pt=new ie,$n=class extends ge{constructor(e){super(),this.vrmLookAt=e,this.type="VRMLookAtQuaternionProxy";const t=this.rotation._onChangeCallback;this.rotation._onChange(()=>{t(),this._applyToLookAt()});const n=this.quaternion._onChangeCallback;this.quaternion._onChange(()=>{n(),this._applyToLookAt()})}_applyToLookAt(){pt.setFromQuaternion(this.quaternion,Aa.EULER_ORDER),this.vrmLookAt.yaw=Qn*pt.y,this.vrmLookAt.pitch=Qn*pt.x}};function Ea(e,t,n){var i,r;const o=new Map,a=new Map;for(const[u,s]of e.humanoidTracks.rotation.entries()){const l=(i=t.getNormalizedBoneNode(u))==null?void 0:i.name;if(l!=null){const d=new wr(`${l}.quaternion`,s.times,s.values.map((c,p)=>n==="0"&&p%2===0?-c:c));a.set(u,d)}}for(const[u,s]of e.humanoidTracks.translation.entries()){const l=(r=t.getNormalizedBoneNode(u))==null?void 0:r.name;if(l!=null){const d=e.restHipsPosition.y,p=t.normalizedRestPose.hips.position[1]/d,h=s.clone();h.values=h.values.map((_,g)=>(n==="0"&&g%3!==1?-_:_)*p),h.name=`${l}.position`,o.set(u,h)}}return{translation:o,rotation:a}}function Pa(e,t){const n=new Map,i=new Map;for(const[r,o]of e.expressionTracks.preset.entries()){const a=t.getExpressionTrackName(r);if(a!=null){const u=o.clone();u.name=a,n.set(r,u)}}for(const[r,o]of e.expressionTracks.custom.entries()){const a=t.getExpressionTrackName(r);if(a!=null){const u=o.clone();u.name=a,i.set(r,u)}}return{preset:n,custom:i}}function La(e,t){if(e.lookAtTrack==null)return null;const n=e.lookAtTrack.clone();return n.name=t,n}function ba(e,t){const n=[],i=Ea(e,t.humanoid,t.meta.metaVersion);if(n.push(...i.translation.values()),n.push(...i.rotation.values()),t.expressionManager!=null){const r=Pa(e,t.expressionManager);n.push(...r.preset.values()),n.push(...r.custom.values())}if(t.lookAt!=null){let r=t.scene.children.find(a=>a instanceof $n);r==null?(console.warn("createVRMAnimationClip: VRMLookAtQuaternionProxy is not found. Creating a new one automatically. To suppress this warning, create a VRMLookAtQuaternionProxy manually"),r=new $n(t.lookAt),r.name="VRMLookAtQuaternionProxy",t.scene.add(r)):r.name===""&&(console.warn("createVRMAnimationClip: VRMLookAtQuaternionProxy is found but its name is not set. Setting the name automatically. To suppress this warning, set the name manually"),r.name="VRMLookAtQuaternionProxy");const o=La(e,`${r.name}.quaternion`);o!=null&&n.push(o)}return new xr("Clip",e.duration,n)}var Ia=class{constructor(){this.duration=0,this.restHipsPosition=new v,this.humanoidTracks={translation:new Map,rotation:new Map},this.expressionTracks={preset:new Map,custom:new Map},this.lookAtTrack=null}};function Zn(e,t){const n=e.length,i=[];let r=[],o=0;for(let a=0;a<n;a++){const u=e[a];o<=0&&(o=t,r=[],i.push(r)),r.push(u),o--}return i}var Ca=new J,Ae=new v,ft=new S,Jn=new S,Na=new S,Oa=new Set(["1.0","1.0-draft"]),Ua=new Set(Object.values(_a)),Va=class{constructor(e){this.parser=e}get name(){return"VRMC_vrm_animation"}afterRoot(e){return jn(this,null,function*(){var t,n,i;const r=e.parser.json,o=r.extensionsUsed;if(o==null||o.indexOf(this.name)==-1)return;const a=(t=r.extensions)==null?void 0:t[this.name];if(a==null)return;const u=a.specVersion;if(u==null)console.warn("VRMAnimationLoaderPlugin: specVersion of the VRMA is not defined. Consider updating the animation file. Assuming the spec version is 1.0.");else{if(!Oa.has(u)){console.warn(`VRMAnimationLoaderPlugin: Unknown VRMC_vrm_animation spec version: ${u}`);return}u==="1.0-draft"&&console.warn("VRMAnimationLoaderPlugin: Using a draft spec version: 1.0-draft. Some behaviors may be different. Consider updating the animation file.")}const s=this._createNodeMap(a),l=yield this._createBoneWorldMatrixMap(e,a),d=(i=(n=a.humanoid)==null?void 0:n.humanBones.hips)==null?void 0:i.node,c=d!=null?yield e.parser.getDependency("node",d):null,p=new v;c==null||c.getWorldPosition(p),p.y<.001&&console.warn("VRMAnimationLoaderPlugin: The loaded VRM Animation might violate the VRM T-pose (The y component of the rest hips position is approximately zero or below.)");const _=e.animations.map((g,m)=>{const f=r.animations[m],x=this._parseAnimation(g,f,s,l);return x.restHipsPosition=p,x});e.userData.vrmAnimations=_})}_createNodeMap(e){var t,n,i,r,o;const a=new Map,u=new Map,s=(t=e.humanoid)==null?void 0:t.humanBones;s&&Object.entries(s).forEach(([p,h])=>{const _=h==null?void 0:h.node;_!=null&&a.set(_,p)});const l=(n=e.expressions)==null?void 0:n.preset;l&&Object.entries(l).forEach(([p,h])=>{const _=h==null?void 0:h.node;_!=null&&u.set(_,p)});const d=(i=e.expressions)==null?void 0:i.custom;d&&Object.entries(d).forEach(([p,h])=>{const{node:_}=h;u.set(_,p)});const c=(o=(r=e.lookAt)==null?void 0:r.node)!=null?o:null;return{humanoidIndexToName:a,expressionsIndexToName:u,lookAtIndex:c}}_createBoneWorldMatrixMap(e,t){return jn(this,null,function*(){var n,i;e.scene.updateWorldMatrix(!1,!0);const r=yield e.parser.getDependencies("node"),o=new Map;if(t.humanoid==null)return o;for(const[a,u]of Object.entries(t.humanoid.humanBones)){const s=u==null?void 0:u.node;if(s!=null){const l=r[s];o.set(a,l.matrixWorld),a==="hips"&&o.set("hipsParent",(i=(n=l.parent)==null?void 0:n.matrixWorld)!=null?i:Ca)}}return o})}_parseAnimation(e,t,n,i){const r=e.tracks,o=t.channels,a=new Ia;return a.duration=e.duration,o.forEach((u,s)=>{const{node:l,path:d}=u.target,c=r[s];if(l==null)return;const p=n.humanoidIndexToName.get(l);if(p!=null){let _=Gn[p];for(;_!=null&&i.get(_)==null;)_=Gn[_];if(_==null&&(_="hipsParent"),d==="translation")if(p!=="hips")console.warn(`The loading animation contains a translation track for ${p}, which is not permitted in the VRMC_vrm_animation spec. ignoring the track`);else{const g=i.get("hipsParent"),m=Zn(c.values,3).flatMap(x=>Ae.fromArray(x).applyMatrix4(g).toArray()),f=c.clone();f.values=new Float32Array(m),a.humanoidTracks.translation.set(p,f)}else if(d==="rotation"){const g=i.get(p),m=i.get(_);g.decompose(Ae,ft,Ae),ft.invert(),m.decompose(Ae,Jn,Ae);const f=Zn(c.values,4).flatMap(T=>Na.fromArray(T).premultiply(Jn).multiply(ft).toArray()),x=c.clone();x.values=new Float32Array(f),a.humanoidTracks.rotation.set(p,x)}else throw new Error(`Invalid path "${d}"`);return}const h=n.expressionsIndexToName.get(l);if(h!=null){if(d==="translation"){const _=c.times,g=new Float32Array(c.values.length/3);for(let f=0;f<g.length;f++)g[f]=c.values[3*f];const m=new yr(`${h}.weight`,_,g);Ua.has(h)?a.expressionTracks.preset.set(h,m):a.expressionTracks.custom.set(h,m)}else throw new Error(`Invalid path "${d}"`);return}if(l===n.lookAtIndex)if(d==="rotation")a.lookAtTrack=c;else throw new Error(`Invalid path "${d}"`)}),a}};/*!
 * @pixiv/three-vrm-core v3.5.5
 * The implementation of core features of VRM, for @pixiv/three-vrm
 *
 * Copyright (c) 2019-2026 pixiv Inc.
 * @pixiv/three-vrm-core is distributed under MIT License
 * https://github.com/pixiv/three-vrm/blob/release/LICENSE
 */const ka=Object.freeze({lacia:{hair:"#c8cad8",eyes:"#8fa6ff",top:"#e8eaf2",accent:"#7d8bff"},snowdrop:{hair:"#c0e2c2",eyes:"#2f9b6b",top:"#f4f4ee",accent:"#7fd6a0"},kouka:{hair:"#c4202f",eyes:"#c02838",top:"#2a2a33",accent:"#ff4a5a"},saturnus:{hair:"#e2d09c",eyes:"#a8823a",top:"#24345a",accent:"#d8e05a"},methode:{hair:"#f2d8b4",eyes:"#d0603a",top:"#2c2840",accent:"#ff9a3c"}}),Ni="./",Da=`${Ni}hie/vrm/hie-base.vrm`,Ba=`${Ni}hie/vrm/idle.vrma`;function Fa(){const e=new Wr;return e.register(t=>new Xs(t)),e.register(t=>new Va(t)),e}const yt=(e,t,n)=>.2126*e+.7152*t+.0722*n;function Kn(e,t,n){const i=e==null?void 0:e.image;if(!(i!=null&&i.width))return e;const r=`${e.uuid}|${t}`;if(n.has(r))return n.get(r);const o=document.createElement("canvas");o.width=i.width,o.height=i.height;const a=o.getContext("2d",{willReadFrequently:!0});a.drawImage(i,0,0);const u=a.getImageData(0,0,o.width,o.height),s=u.data;let l=0,d=0;for(let f=0;f<s.length;f+=4)s[f+3]>32&&(l+=yt(s[f],s[f+1],s[f+2]),d++);const c=Math.max(8,d?l/d:128),p=new H(t),h=p.r*255,_=p.g*255,g=p.b*255;for(let f=0;f<s.length;f+=4){const x=Math.pow(yt(s[f],s[f+1],s[f+2])/c,.85);s[f]=Math.min(255,h*x),s[f+1]=Math.min(255,_*x),s[f+2]=Math.min(255,g*x)}a.putImageData(u,0,0);const m=e.clone();return m.source=new Tr(o),m.needsUpdate=!0,n.set(r,m),m}const Ha=[[/_HAIR|HairBack/i,"hair"],[/EyeIris/i,"eyes"],[/_Tops_/i,"top"]];function Wa(e,t){const n=ka[t];if(!n)return;const i=new Map,r=new H(n.accent).multiplyScalar(.3);e.scene.traverse(o=>{var a,u;if(o.isMesh)for(const s of Array.isArray(o.material)?o.material:[o.material]){if(!s)continue;s.isMToonMaterial&&!/_SKIN|_FACE|_EYE/i.test(s.name||"")&&((a=s.parametricRimColorFactor)==null||a.copy(r),s.parametricRimFresnelPowerFactor=4,s.parametricRimLiftFactor=.02,s.rimLightingMixFactor=.6);const l=(u=Ha.find(([d])=>d.test(s.name||"")))==null?void 0:u[1];if(!(!l||!n[l])){if(s.map&&(s.map=Kn(s.map,n[l],i)),s.shadeMultiplyTexture&&(s.shadeMultiplyTexture=Kn(s.shadeMultiplyTexture,n[l],i)),s.shadeColorFactor){const d=yt(s.shadeColorFactor.r,s.shadeColorFactor.g,s.shadeColorFactor.b);s.shadeColorFactor.copy(new H(n[l]).multiplyScalar(l==="eyes"?1:Math.min(1,.35+d*.6)))}s.needsUpdate=!0}}})}async function za(e,{onProgress:t}={}){var s,l;const n=Fa(),[i,r]=await Promise.all([n.loadAsync(Da,t),n.loadAsync(Ba).catch(()=>null)]),o=i.userData.vrm;if(!o)throw new Error("the file is not a VRM");Z.removeUnnecessaryVertices(i.scene),Z.combineSkeletons(i.scene),Z.rotateVRM0(o),o.scene.traverse(d=>{d.frustumCulled=!1}),Wa(o,e);const a=(l=(s=r==null?void 0:r.userData)==null?void 0:s.vrmAnimations)==null?void 0:l[0],u=a?ba(a,o):null;return{vrm:o,clip:u}}function ei(e){e!=null&&e.scene&&Z.deepDispose(e.scene)}const ti=["neck","head","spine","chest"],ja=new S,Ga=new ie,ni=new v,ii=new v,_e=1.75;function ri(e,t,n){e.position.set(t,n-.1,_e),e.lookAt(t,n-.2,0)}function Xa(e){var i;const t=e.expressionManager,n={};if(!t)return n;for(const r of[...Rt,...Pe,"blink"])t.getExpression(r)?n[r]=r:n[r]=((i=t.expressions.find(o=>o.expressionName.toLowerCase()===r))==null?void 0:i.expressionName)||null;return n}function qa(e){const t=(n,i,r=0)=>{var o;return(o=e.humanoid.getNormalizedBoneNode(n))==null?void 0:o.rotation.set(r,0,i)};t("leftUpperArm",-1.2),t("rightUpperArm",1.2),t("leftLowerArm",-.15),t("rightLowerArm",.15)}function Ya({id:e,drive:t,still:n,onReady:i,onError:r}){const[o,a]=U.useState(null),{camera:u,invalidate:s}=Fr(),l=U.useMemo(()=>new ge,[]),d=U.useRef({face:{},mouth:{},head:[0,0,0],gaze:[0,0],blinkAt:1500,blinkStart:-1,t:0,frames:0,fpsT:0});return U.useEffect(()=>{let c=!1,p=null;return za(e,{onProgress:h=>{t.current.progress=h.total?h.loaded/h.total:0}}).then(h=>{if(c){ei(h.vrm);return}p=h;const _=Object.fromEntries(ti.map(f=>{var x;return[f,(x=h.vrm.humanoid.getNormalizedBoneNode(f))==null?void 0:x.quaternion.clone()]})),g=h.clip?new Rr(h.vrm.scene):null;g?(g.clipAction(h.clip).play(),n&&g.setTime(1)):qa(h.vrm),h.vrm.lookAt.target=l,h.vrm.scene.updateMatrixWorld(!0);const m=h.vrm.humanoid.getNormalizedBoneNode("head").getWorldPosition(new v);t.current.headY=m.y,d.current.camX=m.x,ri(u,m.x,m.y),l.position.set(0,m.y,_e),a({...h,rest:_,mixer:g,names:Xa(h.vrm)}),i==null||i(),s()}).catch(h=>{c||r==null||r(h)}),()=>{c=!0,p&&ei(p.vrm)}},[e]),U.useEffect(()=>{if(!n||!o)return;const c=setInterval(()=>{t.current.busyUntil>performance.now()&&s()},1e3/12);return()=>clearInterval(c)},[n,o,t,s]),Hr(({gl:c},p)=>{var X,V,K,te;if(!o)return;const{vrm:h,mixer:_,rest:g}=o,m=t.current,f=d.current,x=performance.now(),T=n?1/12:Math.min(p,.1);f.t+=T;for(const k of ti)g[k]&&h.humanoid.getNormalizedBoneNode(k).quaternion.copy(g[k]);_==null||_.update(n?0:T);const y=Gr(m.emotion);m.faceOverride&&(y.face={...y.face,...m.faceOverride});const w=m.talk?(x-m.talk.start)/1e3:-1,R=!!m.talk&&w<m.talk.timeline.duration;f.face=on(f.face,qr(y.face,R),T,7),f.mouth=on(f.mouth,R?no(m.talk.timeline,w):{},T,24);const A=h.expressionManager,P=o.names;for(const k of Rt)P[k]&&A.setValue(P[k],f.face[k]||0);for(const k of Pe)P[k]&&A.setValue(P[k],Math.min(1,f.mouth[k]||0));if(!n){f.blinkStart<0&&f.t*1e3>f.blinkAt&&(f.blinkStart=f.t*1e3);const k=f.blinkStart>=0?Qr(f.t*1e3-f.blinkStart):0;f.blinkStart>=0&&k===0&&f.t*1e3-f.blinkStart>100&&(f.blinkStart=-1,f.blinkAt=f.t*1e3+Yr()),P.blink&&A.setValue(P.blink,k*(1-.6*(f.face.happy||0)))}f.gaze=sn(f.gaze,y.gaze,T,4);const N=m.headY||1.3,b=m.focus==="viewer"||R,C=b?0:(((X=m.pointer)==null?void 0:X.x)||0)*.55,D=b?0:(((V=m.pointer)==null?void 0:V.y)||0)*.35;ni.set(C+f.gaze[0],N+D+f.gaze[1],_e),l.position.lerp(ni,1-Math.exp(-(b?6:4)*T)),f.head=sn(f.head,y.head,T,5);const W=Math.atan2(l.position.x,_e)*.35,G=-Math.atan2(l.position.y-N,_e)*.3,M=R?Math.sin(w*6.5)*.025+Math.sin(w*2.3)*.015:0,L=n?0:Math.sin(f.t*1.7)*.012,B=n?0:Math.sin(f.t*.6)*.015,z=(k,Le,be,Ie)=>{const Me=h.humanoid.getNormalizedBoneNode(k);Me&&Me.quaternion.multiply(ja.setFromEuler(Ga.set(Le,be,Ie)))};z("neck",(f.head[0]+G)*.5+M*.5,(f.head[1]+W)*.5,f.head[2]*.5),z("head",(f.head[0]+G)*.5+M,(f.head[1]+W)*.5,f.head[2]*.5+B*.5),z("chest",L,0,B*.4),n?(h.humanoid.update(),(K=h.lookAt)==null||K.update(T),(te=h.expressionManager)==null||te.update()):h.update(T),h.humanoid.getNormalizedBoneNode("head").getWorldPosition(ii),f.camX+=(ii.x-f.camX)*(1-Math.exp(-.8*T)),ri(u,f.camX,N),f.frames++,x-f.fpsT>1e3&&(m.fps=Math.round(f.frames*1e3/(x-f.fpsT||1)),f.frames=0,f.fpsT=x),m.face=f.face,m.mouth=f.mouth,m.talking=R,m.lookAt=[l.position.x,l.position.y],m.calls=c.info.render.calls,m.triangles=c.info.render.triangles}),o?E.jsx("primitive",{object:o.vrm.scene}):null}function Qa({id:e,drive:t,still:n,dpr:i,onReady:r,onError:o}){return E.jsxs(Br,{className:"v3d__canvas",dpr:i,frameloop:n?"demand":"always",gl:{antialias:!n,alpha:!0,powerPreference:"high-performance",preserveDrawingBuffer:!0},camera:{fov:24,near:.05,far:20,position:[0,1.3,_e]},flat:!0,children:[E.jsx("hemisphereLight",{args:["#fff6ee","#2a2a33",.6]}),E.jsx("directionalLight",{position:[-1.2,2.2,2.4],intensity:1.35,color:"#fff1e2"}),E.jsx("directionalLight",{position:[1.4,1.8,-2.2],intensity:.9,color:"#d6ddff"}),E.jsx(Ya,{id:e,drive:t,still:n,onReady:r,onError:o})]})}const $a="The visitor opened a 3D preview of you: a quiet test room where they see you face to face as a 3D figure. Keep replies short and spoken; one (action) at most.";class Za extends U.Component{constructor(t){super(t),this.state={failed:!1}}static getDerivedStateFromError(){return{failed:!0}}componentDidCatch(t){var n,i;(i=(n=this.props).onError)==null||i.call(n,t)}render(){return this.state.failed?null:this.props.children}}function Ja(e){const[t,n]=U.useState(0);return U.useEffect(()=>{if(!e){n(0);return}let i;const r=()=>{const o=io(e.timeline,(performance.now()-e.start)/1e3);n(a=>a===o?a:o),o<e.speech.length&&(i=requestAnimationFrame(r))};return r(),()=>cancelAnimationFrame(i)},[e]),e?e.speech.slice(0,t):""}function sl(){const{id:e}=Sr();return Ar.includes(e)?E.jsx(Ka,{id:e},e):E.jsx(Er,{to:"/",replace:!0})}function Ka({id:e}){var G;const t=U.useRef(null),n=nn.tier||"high",i=(n==="software"||!!nn.softwareRenderer)&&!/[?&]vrm=live\b/.test(window.location.search),r=U.useRef({emotion:"default",talk:null,focus:"pointer",pointer:{x:0,y:0},busyUntil:0,progress:0}),[o,a]=U.useState("loading"),[u,s]=U.useState("default"),[l,d]=U.useState(null),[c,p]=U.useState(""),h=U.useRef("default"),_=Pr[e].character_name,g=Dr(e),m=U.useCallback(()=>Lr(br()),[]);U.useLayoutEffect(()=>{document.documentElement.classList.add("is-hie3d");const M=Ir(t.current,()=>m(),"hie3d");return()=>{M(),document.documentElement.classList.remove("is-hie3d")}},[m]);const f=U.useCallback(M=>{const L=li(M);r.current.emotion=L,r.current.busyUntil=Math.max(r.current.busyUntil,performance.now()+1500),s(L)},[]),x=U.useCallback((M,L,B=!1)=>{const z=B?M:eo(M),X=B?[]:[...String(M).matchAll(/\(([^()]*)\)|\*([^*\n]+)\*/g)].map(k=>k[1]||k[2]),V=to(B?"":z),K=performance.now();r.current.talk=B?null:{timeline:V,start:K},r.current.busyUntil=K+V.duration*1e3+800;const te=L||(B?null:rn(M))||(r.current.emotion==="thinking"?h.current:null);te&&f(te),d({content:M,speech:z,acts:X,timeline:V,start:K,notice:B})},[f]),T=ue(M=>{var L;return(L=M.chars[g])==null?void 0:L.messages}),y=ue(M=>!!M.busy[g]),w=ue(M=>!Cr(M)),R=ue(M=>{var L;return(L=M.chars[g])==null?void 0:L.suggested})||[];U.useEffect(()=>{const M=ue.getState();M.init();const L=Nr[e];M.stageIntro(e,{sceneId:"preview3d",scene:$a,lines:[{text:L.content,emotion:rn(L.content)||"happy"}],cards:L.replies})},[e]);const A=U.useRef(null);U.useEffect(()=>{const M=[...T||[]].reverse().find(L=>L.role==="assistant");!M||M.id===A.current||(A.current=M.id,x(M.content,M.system?null:M.emotion,!!M.system))},[T,x]),U.useEffect(()=>{y?(h.current=r.current.emotion,f("thinking"),r.current.focus="viewer"):r.current.focus="pointer"},[y,f]);const P=M=>{const L=(M??c).trim();!L||y||(p(""),ue.getState().send(g,L))};U.useEffect(()=>(window.__vrm={state:()=>{const M=r.current;return{id:e,status:o,tier:n,still:i,emotion:M.emotion,face:{...M.face},mouth:{...M.mouth},talking:!!M.talking,fps:M.fps||0,lookAt:M.lookAt,calls:M.calls,triangles:M.triangles}},emotion:M=>f(M),say:(M,L)=>x(M,L),face:M=>{r.current.faceOverride=M||null,r.current.busyUntil=performance.now()+1500}},()=>{delete window.__vrm}),[e,o,n,i,f,x]);const N=Ja(l!=null&&l.notice?null:l),b=Or(e).character_meta_info.emotion_images.default,C=M=>{const L=t.current.getBoundingClientRect();r.current.pointer={x:(M.clientX-L.left)/L.width*2-1,y:-((M.clientY-L.top)/L.height*2-1)},r.current.busyUntil=Math.max(r.current.busyUntil,performance.now()+1200)},D=M=>{p(M),r.current.focus=M||y?"viewer":"pointer"},W=!y&&!w?R.slice(0,3):[];return Ur.createPortal(E.jsxs("div",{className:"v3d",ref:t,"data-screen":"hie3d","data-status":o,"data-emotion":u,role:"dialog","aria-modal":"true","aria-label":`${_}, 3D preview`,onPointerMove:C,"data-lenis-prevent":!0,children:[E.jsx("div",{className:"v3d__bg","aria-hidden":!0}),E.jsx(Za,{onError:()=>a("error"),children:E.jsx(Qa,{id:e,drive:r,still:i,dpr:Vr(n).pixelRatio,onReady:()=>a("ready"),onError:()=>a("error")})}),o==="loading"&&E.jsxs("p",{className:"v3d__loading",role:"status",children:["Loading ",_," in 3D…"]}),o==="error"&&E.jsxs("div",{className:"v3d__fallback",role:"status",children:[E.jsx("img",{src:b,alt:""}),E.jsx("p",{children:"The 3D model could not load on this device. Her portrait stands in."})]}),E.jsxs("header",{className:"v3d__bar",children:[E.jsxs("button",{type:"button",className:"v3d__btn",onClick:m,"data-autofocus":!0,children:[E.jsx("i",{"aria-hidden":!0,children:"←"})," Back"]}),E.jsxs("p",{className:"v3d__name",children:[E.jsx("b",{children:_}),E.jsxs("span",{children:["3D preview · test model",i?" · still (software GL)":""]})]}),E.jsx("span",{className:"v3d__esc","aria-hidden":!0,children:"Esc back"})]}),E.jsx("nav",{className:"v3d__emotions","aria-label":"Expression",children:["default",...kr].map(M=>E.jsx("button",{type:"button",className:"v3d__chip","aria-pressed":u===M,onClick:()=>f(M),children:M==="default"?"neutral":M},M))}),E.jsxs("section",{className:"v3d__talk",children:[E.jsxs("div",{className:"v3d__box","aria-live":"polite",children:[E.jsx("p",{className:"v3d__who",children:l!=null&&l.notice?"Notice":_}),((G=l==null?void 0:l.acts)==null?void 0:G.length)>0&&E.jsx("p",{className:"v3d__act",children:l.acts.join(" · ")}),E.jsx("p",{className:"v3d__text",children:l!=null&&l.notice?l.speech:N}),y&&E.jsxs("p",{className:"v3d__wait","aria-label":`${_} is answering`,children:[E.jsx("span",{}),E.jsx("span",{}),E.jsx("span",{})]})]}),W.length>0&&E.jsx("div",{className:"v3d__cards",children:W.map(M=>E.jsx("button",{type:"button",className:"v3d__card",onClick:()=>P(M),children:M},M))}),E.jsxs("form",{className:"v3d__input",onSubmit:M=>{M.preventDefault(),P()},children:[E.jsx("input",{value:c,onChange:M=>D(M.target.value),placeholder:w?"Free chat is offline here":`Say something to ${_}…`,maxLength:400,"aria-label":"Say",disabled:w}),E.jsx("button",{type:"submit",disabled:y||w||!c.trim(),children:"Say"}),E.jsx("button",{type:"button",onClick:()=>l&&!l.notice&&x(l.content,u),disabled:!l||l.notice,children:"Replay"})]})]})]}),document.body)}export{sl as default};
