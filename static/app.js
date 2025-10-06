async function checkHealth(){
  try{
    const r = await fetch('/api/health');
    const j = await r.json();
    const el = document.getElementById('health');
    if(j.index_loaded){ el.innerHTML = '<span class="ok">Index loaded</span>'; }
    else if(j.index_exists){ el.innerHTML = '<span class="warn">Index exists (not loaded yet)</span>'; }
    else { el.innerHTML = '<span class="warn">No index yet — upload & build</span>'; }
  }catch{
    document.getElementById('health').innerHTML = '<span class="err">Server unreachable</span>';
  }
}

async function listFiles(){
  const r = await fetch('/api/list'); const j = await r.json();
  const el = document.getElementById('filelist');
  el.textContent = j.files.map(f=>`${f.path}  (${f.bytes} bytes)`).join('\n');
}

async function doUpload(){
  const inp = document.getElementById('files');
  if(!inp.files.length){ alert('Select files first'); return; }
  const fd = new FormData();
  for(const f of inp.files){ fd.append('files', f, f.name); }
  const btn = document.getElementById('upload'); btn.disabled = true; btn.textContent = 'Uploading…';
  try{
    const r = await fetch('/api/upload', {method:'POST', body:fd});
    if(!r.ok){ const txt = await r.text(); throw new Error(txt || ('HTTP '+r.status)); }
    const j = await r.json();
    await listFiles();
    alert(`Saved ${j.saved.length} file(s) to DATA_DIR`);
  }catch(e){ alert('Upload failed: '+e); }
  finally{ btn.disabled = false; btn.textContent = 'Upload'; }
}

async function clearData(){
  if(!confirm('Delete all files in DATA_DIR?')) return;
  await fetch('/api/data', {method:'DELETE'});
  document.getElementById('filelist').textContent='';
  alert('DATA_DIR cleared.');
}

async function buildIndex(){
  const chunk = Number(document.getElementById('chunk').value||1200);
  const overlap = Number(document.getElementById('overlap').value||200);
  const btn = document.getElementById('build'); btn.disabled = true; btn.textContent = 'Building…';
  const out = document.getElementById('buildout'); out.textContent='';
  try{
    const fd = new FormData(); fd.append('chunk_size', chunk); fd.append('overlap', overlap);
    const r = await fetch('/api/build', {method:'POST', body: fd});
    if(!r.ok){ throw new Error(await r.text()); }
    const j = await r.json();
    out.textContent = `OK — docs: ${j.stats.docs}, chunks: ${j.stats.chunks}, dim: ${j.stats.dim}, time: ${j.ms}ms`;
    await checkHealth();
  }catch(e){ out.textContent = 'Build failed: '+e; }
  finally{ btn.disabled = false; btn.textContent = 'Build'; }
}

async function ask(){
  const q = document.getElementById('q').value.trim();
  const k = Number(document.getElementById('k').value || 5);
  if(!q){ alert('Please enter a question'); return; }
  const btn = document.getElementById('ask');
  btn.disabled = true; btn.textContent = 'Thinking…';
  document.getElementById('ragAns').textContent = '';
  document.getElementById('llmAns').textContent = '';
  document.getElementById('chunks').innerHTML = '';
  document.getElementById('meta').textContent = '';
  document.getElementById('timing').innerHTML = '';

  try{
    const r = await fetch('/api/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q, top_k:k})});
    if(!r.ok){ throw new Error(await r.text()); }
    const j = await r.json();
    document.getElementById('ragAns').textContent = j.rag.answer || '';
    document.getElementById('llmAns').textContent = j.llm.answer || '';

    const chunks = j.rag.chunks || [];
    const chunksEl = document.getElementById('chunks');
    for(const c of chunks){
      const div = document.createElement('div');
      const header = `[${(c.source||'').split('/').slice(-1)[0]}#chunk${c.chunk_id}] (score=${(c.score||0).toFixed(3)})`;
      /* use mono-scroll for chunk text so long lines don't blow the card */
      div.innerHTML = `<div class="chip">${header}</div><div class="mono-scroll">${(c.text||'').slice(0,1200)}</div>`;
      chunksEl.appendChild(div);
    }

    const used = j.used || {}; const meta = document.getElementById('meta');
    meta.textContent = `Model: ${used.model} | Embed: ${used.embed} | Top-K: ${used.top_k}`;

    const t = j.latency_ms || {}; const timing = document.getElementById('timing');
    for(const [k,v] of Object.entries(t)){
      const span = document.createElement('span');
      span.className = 'chip'; span.textContent = `${k}: ${v}ms`;
      timing.appendChild(span);
    }
  }catch(err){
    document.getElementById('ragAns').textContent = 'Error: ' + err.message;
  }finally{
    btn.disabled = false; btn.textContent = 'Ask';
  }
}

document.getElementById('upload').addEventListener('click', doUpload);
document.getElementById('clear').addEventListener('click', clearData);
document.getElementById('build').addEventListener('click', buildIndex);
document.getElementById('ask').addEventListener('click', ask);
window.addEventListener('keydown', (e)=>{ if(e.key==='Enter' && (e.metaKey||e.ctrlKey)) ask(); });

checkHealth();
listFiles();
