/* 🔬 stock_lab.js — باك تست walk-forward لمحرك توصيات mistral-Mi (الأسهم المتوافقة مع الشريعة فقط ☪️)
   الفكرة: نفس محرك التطبيق (analyze) المستخرج من index.html + تاريخ hist.json
   المقارنة: (0) الأساس الحالي · (1) + فلتر حالة السوق (اتساع ≥50% فوق SMA50) · (2) + وقف ATR أوسع · (3) الاثنين معًا */
const fs=require('fs');
const hist=JSON.parse(fs.readFileSync('hist.json','utf8'));
const SYMS=Object.keys(hist.stocks).filter(s=>SHARIA.includes(s));
const bySym={},idxOf={};
let allDates=new Set();
SYMS.forEach(s=>{
  const arr=(hist.stocks[s]||[]).filter(c=>c&&c.c>0).map(c=>({d:c.d,o:(c.o!=null?c.o:c.c),h:(c.h!=null?c.h:c.c),l:(c.l!=null?c.l:c.c),c:c.c,v:(c.v||0)}));
  bySym[s]=arr;const m={};arr.forEach((c,i)=>{m[c.d]=i;allDates.add(c.d)});idxOf[s]=m;
});
const days=Array.from(allDates).sort();
function smaN(arr,i,n){if(i+1<n)return null;let s=0;for(let k=i-n+1;k<=i;k++)s+=arr[k].c;return s/n;}
/* اتساع السوق: نسبة الأسهم فوق SMA50 — مقياس حالة السوق من غير ما نحتاج بيانات المؤشر */
const breadth={};
days.forEach(d=>{let above=0,tot=0;SYMS.forEach(s=>{const i=idxOf[s][d];if(i==null||i<50)return;tot++;const a=smaN(bySym[s],i,50);if(a&&bySym[s][i].c>a)above++;});breadth[d]=tot?(above/tot*100).toFixed(1):0;});
function atr14(arr,i){if(i<14)return null;let s=0;for(let k=i-13;k<=i;k++){const tr=Math.max(arr[k].h-arr[k].l,Math.abs(arr[k].h-arr[k-1].c),Math.abs(arr[k].l-arr[k-1].c));s+=tr;}return s/14;}
function liqOK(arr,i){const v=arr[i].v;return v==null||arr[i].c*v>=1000000;}
/* محاكاة: دخول عند إغلاق يوم الإشارة (sc>=1.5) · خروج: وقف / هدف / إشارة بيع (sc<=-1.5) / 20 جلسة · حد أقصى 5 مراكز (زي الروبوت) */
function runBT(opts){
  const pos={},trades=[];
  for(let di=0;di<days.length;di++){
    const d=days[di];
    Object.keys(pos).forEach(s=>{
      const p=pos[s],i=idxOf[s][d];if(i==null)return;
      const c=bySym[s][i].c;p.daysHeld++;
      let exit=null,ret=null;
      if(c<=p.stop){exit='وقف';ret=(c-p.entry)/p.entry*100;}
      else if(c>=p.t1){exit='هدف';ret=(c-p.entry)/p.entry*100;}
      else if(p.daysHeld>=20){exit='زمن';ret=(c-p.entry)/p.entry*100;}
      else{const v=analyze(bySym[s].slice(0,i+1));if(v&&v.sc<=-1.5){exit='إشارة';ret=(c-p.entry)/p.entry*100;}}
      if(exit){trades.push({ret:ret,days:p.daysHeld,why:exit});delete pos[s];}
    });
    if(di>=days.length-1)continue;
    SYMS.forEach(s=>{
      if(pos[s])return;const i=idxOf[s][d];if(i==null||i<60)return;
      if(Object.keys(pos).length>=5)return;
      const arr=bySym[s];if(!liqOK(arr,i))return;
      if(opts.regime&&parseFloat(breadth[d])<50)return;
      const v=analyze(arr.slice(0,i+1));if(!v||v.sc<1.5)return;
      let stop=v.stop;
      if(opts.atr){const a=atr14(arr,i);if(a)stop=Math.min(v.stop,arr[i].c-2.2*a);}
      if(stop>=arr[i].c)return;
      pos[s]={entry:arr[i].c,stop:stop,t1:v.t1,daysHeld:0};
    });
  }
  const n=trades.length;if(!n)return{trades:0};
  const wins=trades.filter(t=>t.ret>0).length;
  const gp=trades.filter(t=>t.ret>0).reduce((a,t)=>a+t.ret,0);
  const gl=Math.abs(trades.filter(t=>t.ret<0).reduce((a,t)=>a+t.ret,0));
  const avg=trades.reduce((a,t)=>a+t.ret,0)/n;
  const why={};trades.forEach(t=>{why[t.why]=(why[t.why]||0)+1});
  return{trades:n,win:(wins/n*100).toFixed(1),pf:gl>0?(gp/gl).toFixed(2):'∞',avg:+avg.toFixed(2),avgDays:+(trades.reduce((a,t)=>a+t.days,0)/n).toFixed(1),why:why};
}
const res={base:runBT({}),regime:runBT({regime:true}),atr:runBT({atr:true}),combo:runBT({regime:true,atr:true})};
const out={t:Date.now(),days:days.length,syms:SYMS.length,first:days[0],last:days[days.length-1],breadthLast:breadth[days[days.length-1]],res};
fs.writeFileSync('stock_bt.json',JSON.stringify(out,null,1));
const row=k=>{const r=res[k];return k.padEnd(8)+' صفقات:'+((''+r.trades).padStart(4))+' نجاح:'+((''+r.win).padStart(6))+'٪ PF:'+((''+r.pf).padStart(6))+' متوسط:'+((''+r.avg).padStart(7))+'٪ جلسات:'+(r.avgDays||'-')};
console.log('🔬 باك تست الأسهم المتوافقة مع الشريعة — '+SYMS.length+' سهم · '+days.length+' جلسة ('+days[0]+' ← '+days[days.length-1]+')');
console.log('اتساع السوق الحالي: '+breadth[days[days.length-1]]+'٪ فوق SMA50');
console.log(row('base'));
console.log(row('regime'));
console.log(row('atr'));
console.log(row('combo'));
