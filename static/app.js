const input = document.querySelector('#fileInput');
const dropzone = document.querySelector('#dropzone');
const processButton = document.querySelector('#processButton');
const downloadButton = document.querySelector('#downloadButton');
const statusBox = document.querySelector('#status');
const statusTitle = document.querySelector('#statusTitle');
const statusMessage = document.querySelector('#statusMessage');
const errorBox = document.querySelector('#error');
const dropTitle = document.querySelector('#dropTitle');
const dropHint = document.querySelector('#dropHint');
const rightsCheckbox = document.querySelector('#rightsCheckbox');
const downloadNotice = document.querySelector('#downloadNotice');
let selectedFile = null;
const API_BASE = (window.SAMPLESPLIT_API_BASE || '').replace(/\/$/, '');
const PROCESSING_TIMEOUT_MS = 30 * 60 * 1000;

function apiUrl(path){ return `${API_BASE}${path}`; }
function updateButton(){ processButton.disabled = !(selectedFile && rightsCheckbox.checked); }

function showError(message){ errorBox.textContent=message; errorBox.hidden=false; }
function choose(file){
  errorBox.hidden=true; downloadButton.hidden=true;
  if(!file) return;
  const ext=file.name.toLowerCase().split('.').pop();
  if(!['mp3','wav'].includes(ext)){showError('Please choose an MP3 or WAV file.');return;}
  if(file.size>100*1024*1024){showError('That file is larger than the 100 MB limit.');return;}
  selectedFile=file; dropTitle.textContent=file.name; dropHint.textContent=`${(file.size/1024/1024).toFixed(1)} MB · ready to process`; updateButton();
}
dropzone.addEventListener('click',()=>input.click());
dropzone.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();input.click();}});
input.addEventListener('change',()=>choose(input.files[0]));
rightsCheckbox.addEventListener('change',updateButton);
['dragenter','dragover'].forEach(name=>dropzone.addEventListener(name,e=>{e.preventDefault();dropzone.classList.add('dragging');}));
['dragleave','drop'].forEach(name=>dropzone.addEventListener(name,e=>{e.preventDefault();dropzone.classList.remove('dragging');}));
dropzone.addEventListener('drop',e=>choose(e.dataTransfer.files[0]));

async function readError(response){try{return (await response.json()).detail;}catch{return 'Something went wrong. Please try again.';}}
function connectionError(error){
  console.error('SampleSplit API request failed', {name:error?.name, message:error?.message, apiBase:API_BASE || 'same-origin'});
  if(error?.name==='AbortError') return 'Processing timed out.';
  if(!navigator.onLine) return 'Your Mac appears to be offline.';
  return 'Audio processing backend is unavailable. Start SampleSplit in Terminal, then reload this page.';
}
async function request(path, options){
  try{return await fetch(apiUrl(path), options);}catch(error){throw new Error(connectionError(error));}
}
async function poll(jobId){
  const startedAt=Date.now();
  while(true){
    if(Date.now()-startedAt>PROCESSING_TIMEOUT_MS) throw new Error('Processing timed out.');
    const response=await request(`/api/status/${jobId}`);
    if(!response.ok) throw new Error(await readError(response));
    const job=await response.json(); statusMessage.textContent=job.message;
    if(job.status==='complete'){
      statusTitle.textContent='Your sample pack is ready'; statusBox.querySelector('.spinner').style.display='none';
      downloadButton.href=apiUrl(`/api/download/${jobId}`); downloadButton.hidden=false; downloadNotice.hidden=false; processButton.hidden=true; return;
    }
    if(job.status==='error') throw new Error(job.message);
    await new Promise(resolve=>setTimeout(resolve,1500));
  }
}
processButton.addEventListener('click',async()=>{
  if(!selectedFile || !rightsCheckbox.checked)return;
  errorBox.hidden=true; processButton.disabled=true; statusBox.hidden=false; statusTitle.textContent='Working on your track'; statusMessage.textContent='Uploading…';
  const data=new FormData(); data.append('file',selectedFile); data.append('rights_confirmed','true');
  try{
    const controller=new AbortController();
    const uploadTimer=setTimeout(()=>controller.abort(),2*60*1000);
    const response=await request('/api/process',{method:'POST',body:data,signal:controller.signal});
    clearTimeout(uploadTimer);
    if(!response.ok)throw new Error(await readError(response));
    const {job_id}=await response.json(); await poll(job_id);
  }catch(error){statusBox.hidden=true;showError(error.message);updateButton();}
});

request('/api/health').then(async response=>{
  if(!response.ok) throw new Error(await readError(response));
  const health=await response.json();
  if(!health.processing) showError('Audio processing backend is unavailable because required audio tools are missing.');
}).catch(error=>showError(error.message || connectionError(error)));
