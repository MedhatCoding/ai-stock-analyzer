/* mistral-Mi — Service Worker: تثبيت كتطبيق حقيقي + شغل أوفلاين */
const SAHMY_CACHE="mistral-mi-v3.53";
const SAHMY_ASSETS=["./","./index.html","./manifest.json","./icon.svg","./scorecard.html"];
self.addEventListener("install",function(e){
  e.waitUntil(caches.open(SAHMY_CACHE).then(function(c){return c.addAll(SAHMY_ASSETS)}).then(function(){return self.skipWaiting()}));
});
self.addEventListener("activate",function(e){
  e.waitUntil(caches.keys().then(function(ks){return Promise.all(ks.filter(function(k){return k!==SAHMY_CACHE}).map(function(k){return caches.delete(k)}))}).then(function(){return self.clients.claim()}));
});
self.addEventListener("fetch",function(e){
  const url=new URL(e.request.url);
  if(url.origin!==self.location.origin)return;
  if(e.request.mode==="navigate"){
    /* الصفحة الرئيسية: الشبكة الأول عشان التحديثات توصل فورًا، والكاش احتياطي أوفلاين */
    e.respondWith(
      fetch(e.request).then(function(r){
        const cp=r.clone();
        caches.open(SAHMY_CACHE).then(function(c){c.put("./index.html",cp)});
        return r;
      }).catch(function(){return caches.match("./index.html")})
    );
    return;
  }
  /* باقي ملفات الموقع: كاش الأول */
  e.respondWith(
    caches.match(e.request).then(function(m){
      return m||fetch(e.request).then(function(r){
        const cp=r.clone();
        caches.open(SAHMY_CACHE).then(function(c){c.put(e.request,cp)});
        return r;
      }).catch(function(){return m});
    })
  );
});
