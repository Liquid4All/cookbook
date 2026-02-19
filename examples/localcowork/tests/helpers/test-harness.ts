import * as fsNode from "fs/promises";
import * as path from "path";
import * as os from "os";
import { initSandbox } from "../../mcp-servers/_shared/ts/validation";

export interface MCPToolResult { readonly success: boolean; readonly data: unknown; }
interface MCPToolModule { execute(p: Record<string, unknown>): Promise<MCPToolResult>; }

const STP: Record<string, string> = {
  filesystem:"mcp-servers/filesystem/src/tools",data:"mcp-servers/data/src/tools",
  clipboard:"mcp-servers/clipboard/src/tools",system:"mcp-servers/system/src/tools",
  task:"mcp-servers/task/src/tools",calendar:"mcp-servers/calendar/src/tools",
  email:"mcp-servers/email/src/tools",audit:"mcp-servers/audit/src/tools",
};
const TEM: Record<string,string> = {
  list_dir:"listDir",read_file:"readFile",write_file:"writeFile",move_file:"moveFile",
  copy_file:"copyFile",delete_file:"deleteFile",search_files:"searchFiles",get_metadata:"getMetadata",
  write_csv:"writeCsv",deduplicate_records:"deduplicateRecords",summarize_anomalies:"summarizeAnomalies",
  get_clipboard:"getClipboard",set_clipboard:"setClipboard",clipboard_history:"clipboardHistory",
  get_system_info:"getSystemInfo",list_processes:"listProcesses",
  create_task:"createTask",list_tasks:"listTasks",get_overdue:"getOverdue",
  update_task:"updateTask",daily_briefing:"dailyBriefing",
  create_event:"createEvent",list_events:"listEvents",find_free_slots:"findFreeSlots",
  create_time_block:"createTimeBlock",draft_email:"draftEmail",
  get_tool_log:"getToolLog",get_session_summary:"getSessionSummary",
  generate_audit_report:"generateAuditReport",
};
const PR=path.resolve(path.dirname(new URL(import.meta.url).pathname),"..","..");

export class TestHarness {
  tempDir=""; private readonly fsd:string; private ok=false;
  constructor(private readonly ucId:string){this.fsd=path.join(PR,"tests","fixtures",ucId);}
  async setup():Promise<void>{
    this.tempDir=await fsNode.mkdtemp(path.join(os.tmpdir(),"lc-uc-"+this.ucId+"-"));
    initSandbox([os.tmpdir(),"/private/var/folders","/private/tmp","/tmp"]);
    try{await fsNode.access(this.fsd);await cdr(this.fsd,this.tempDir);}catch{}
    this.ok=true;}
  async teardown():Promise<void>{if(this.tempDir)await fsNode.rm(this.tempDir,{recursive:true,force:true});this.ok=false;}
  tempPath(...s:string[]):string{this.c();return path.join(this.tempDir,...s);}
  fixturePath(...s:string[]):string{return path.join(this.fsd,...s);}
  async callTsTool(srv:string,tn:string,par:Record<string,unknown>):Promise<MCPToolResult>{
    this.c();const sp=STP[srv];if(!sp)throw new Error("Unknown srv:"+srv);
    const en=TEM[tn];if(!en)throw new Error("Unknown tool:"+tn);
    const fp=path.join(PR,sp,tn+".ts");
    const m=(await import(fp))as Record<string,MCPToolModule>;const t=m[en];
    if(!t||typeof t.execute!=="function")throw new Error("Not found:"+tn);
    return t.execute(par);}
  async writeTemp(r:string,c:string):Promise<string>{
    this.c();const f=this.tempPath(r);await fsNode.mkdir(path.dirname(f),{recursive:true});
    await fsNode.writeFile(f,c,"utf-8");return f;}
  async readTemp(r:string):Promise<string>{this.c();return fsNode.readFile(this.tempPath(r),"utf-8");}
  async existsTemp(r:string):Promise<boolean>{try{await fsNode.access(this.tempPath(r));return true;}catch{return false;}}
  private c():void{if(!this.ok)throw new Error("Not initialized");}
}
async function cdr(s:string,d:string):Promise<void>{
  for(const e of await fsNode.readdir(s,{withFileTypes:true})){
    const a=path.join(s,e.name),b=path.join(d,e.name);
    if(e.isDirectory()){await fsNode.mkdir(b,{recursive:true});await cdr(a,b);}
    else await fsNode.copyFile(a,b);}}