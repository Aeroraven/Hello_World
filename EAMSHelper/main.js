function eamsHelper(){
	eamsHelper.getPointsReportAddr="(Empty)";
	eamsHelper.getPointsReportCacheAddr="";
    eamsHelper.standardInterval = 200;

    eamsHelper.callbackAddr = "(Empty)";

	this._this=this;
	this.getPointsReportStatus=0;
    this.getPointsReportTimer = 0;

    this.uiProgTimer = 0;

    this.curStuId = 0;
    this.begStuId = 0;
    this.endStuId = 0;

    this.htmlDecode = function(text) {
        var temp = document.createElement("div");
        temp.innerHTML = text;
        var output = temp.innerText || temp.textContent;
        temp = null;
        return output;
    } 

	this.standardXHRGet = function(sUrl,bAsync,fOnSuccess=function(obj){},fOnError=function(errId){}){
		var obj=new XMLHttpRequest();
		var ret;
		var vaild=0;
		obj.onreadystatechange=function(){
			if (obj.readyState==4 && obj.status==200){
				ret=obj.responseText;
				fOnSuccess(obj);
			}else{
				fOnError(obj.status);
			}
		}
		obj.open("GET",sUrl,bAsync);
		obj.send();
	}
    this.pageInject = function (self=this) {
        var injectHost = document.getElementsByTagName("body")[0];
        var injectStr = atob("Jmx0O3NjcmlwdCBzcmM9ImVhbXNIZWxwZXIuanMiJmd0OyZsdDsvc2NyaXB0Jmd0Ow0KDQombHQ7ZGl2IGlkPSJlYW1zSGVscGVyX2luamVjdG9yX21haW4iIHN0eWxlPSJiYWNrZ3JvdW5kLWNvbG9yOiMwMDAwMDA7Y29sb3I6I2ZmZmZmZjtsZWZ0OjBweDtyaWdodDowcHg7dG9wOjBweDtib3R0b206MHB4O3Bvc2l0aW9uOmZpeGVkO21hcmdpbi1sZWZ0OjBweDttYXJnaW4tcmlnaHQ6MHB4O21hcmdpbi10b3A6MHB4O21hcmdpbi1ib3R0b206MHB4O2ZvbnQtc2l6ZToxN3B4OyImZ3Q7DQoJJmx0O3NjcmlwdCZndDsNCgkJLy9FQU1TSGVscGVyIENhbGxlcg0KCQkNCgkmbHQ7L3NjcmlwdCZndDsNCgkmbHQ7ZGl2IGlkPSJlYW1zSGVscGVyX2pzb25wX3BsYWNlbWVudCImZ3Q7DQoJJmx0Oy9kaXYmZ3Q7DQoJJmx0O3N0eWxlJmd0Ow0KCS5lYW1zSGVscGVyX2NsX2V4cGx7DQoJCWNvbG9yOiNhYWFhYWE7DQoJCWZvbnQtc2l6ZToxNHB4Ow0KCQlwYWRkaW5nLWxlZnQ6MTBweDsNCgkJbWFyZ2luLXRvcDo4cHg7DQoJCWRpc3BsYXk6YmxvY2s7DQoJfQ0KCS5lYW1zX2NsX2J0bnsNCgkJZm9udC1zaXplOjE4cHg7DQoJCWJvcmRlci1yYWRpdXM6NHB4Ow0KCQliYWNrZ3JvdW5kLWNvbG9yOiM0MDhkZWE7DQoJCWNvbG9yOiNmZmZmZmY7DQoJCWJvcmRlcjowcHg7DQoJCXBhZGRpbmctdG9wOjEwcHg7DQoJCXBhZGRpbmctYm90dG9tOjEwcHg7DQoJCXBhZGRpbmctbGVmdDoyMHB4Ow0KCQlwYWRkaW5nLXJpZ2h0OjIwcHg7DQoJCW1hcmdpbi1yaWdodDo1cHg7DQoJfQ0KCS5lYW1zSGVscGVyX2NsX2lucHV0ew0KCQlib3JkZXItdG9wOjBweDsNCgkJYm9yZGVyLWxlZnQ6MHB4Ow0KCQlib3JkZXItcmlnaHQ6MHB4Ow0KCQliYWNrZ3JvdW5kLWNvbG9yOiMwMDAwMDA7DQoJCWJvcmRlci1ib3R0b206MnB4ICMyMDZkY2Egc29saWQ7DQoJCWNvbG9yOiNmZmZmZmY7DQoJCWZvbnQtc2l6ZToyMHB4Ow0KCX0NCgkmbHQ7L3N0eWxlJmd0Ow0KCSZsdDtkaXYgaWQ9ImVhbXNIZWxwZXJfdGl0bGUiIHN0eWxlPSJiYWNrZ3JvdW5kLWNvbG9yOiMyMDZkY2E7cGFkZGluZy10b3A6MjVweDtwYWRkaW5nLWJvdHRvbToyNXB4O3BhZGRpbmctbGVmdDoxNXB4O2ZvbnQtc2l6ZToyNXB4O2ZvbnQtZmFtaWx5OmJhaG5zY2hyaWZ0OyImZ3Q7RUFNU0hlbHBlciZsdDsvZGl2Jmd0Ow0KCSZsdDtkaXYgaWQ9ImVhbXNIZWxwZXJfbWFpbiIgc3R5bGU9InBhZGRpbmctbGVmdDo0MHB4O3BhZGRpbmctdG9wOjI1cHgiJmd0Ow0KCQkmbHQ7ZGl2IGlkPSJlYW1zSGVscGVyX3Byb2dCYXIiJmd0Ow0KCQkJJmx0O2RpdiBzdHlsZT0ibWFyZ2luOjAgYXV0bzsiJmd0Ow0KCQkJCSZsdDt0YWJsZSBzdHlsZT0id2lkdGg6OTAlIiZndDsNCgkJCQkJJmx0O3RyIHN0eWxlPSJ3aWR0aDo5MCUiJmd0Ow0KCQkJCQkJJmx0O3RkIHN0eWxlPSJ3aWR0aDo4MCUiJmd0Ow0KCQkJCQkJCSZsdDtkaXYgaWQ9ImVhbXNIZWxwZXJfcHJvZ0Jhcl9NYWluIiBzdHlsZT0ibWFyZ2luLXRvcDoxMHB4O21hcmdpbi1ib3R0b206MTBweDt3aWR0aDoxMDAlO2JhY2tncm91bmQtY29sb3I6I2FhYWFhYTtoZWlnaHQ6NDBweDtkaXNwbGF5OmlubGluZS1ibG9jaztib3JkZXItcmFkaXVzOjVweDsiJmd0Ow0KCQkJCQkJCQkmbHQ7ZGl2IGlkPSJlYW1zSGVscGVyX3Byb2dCYXJfUHJvY2VzcyIgc3R5bGU9InRleHQtYWxpZ246bGVmdDttYXJnaW4tYm90dG9tOjEwcHg7d2lkdGg6MSU7YmFja2dyb3VuZC1jb2xvcjojMjA2ZGNhO2hlaWdodDo0MHB4O2Rpc3BsYXk6aW5saW5lLWJsb2NrO2JvcmRlci1yYWRpdXM6NXB4OyImZ3Q7DQoJCQkJCQkJDQoJCQkJCQkJCSZsdDsvZGl2Jmd0Ow0KCQkJCQkJCSZsdDsvZGl2Jmd0Ow0KCQkJCQkJJmx0Oy90ZCZndDsNCgkJCQkJCSZsdDt0ZCBzdHlsZT0id2lkdGg6NSUiJmd0Ow0KCQkJCQkJCSZsdDtzcGFuIGlkPSJlYW1zSGVscGVyX3Byb2dfcGVyY2VudCIgc3R5bGU9ImNvbG9yOiNmZmZmZmY7Zm9udC13ZWlnaHQ6Ym9sZDtmb250LXNpemU6MjVweDtwYWRkaW5nLWxlZnQ6NXB4OyImZ3Q7MCUmbHQ7L3NwYW4mZ3Q7DQoJCQkJCQkmbHQ7L3RkJmd0Ow0KCQkJCQkmbHQ7L3RyJmd0Ow0KCQkJCSZsdDsvdGFibGUmZ3Q7DQoJCQkJDQoJCQkmbHQ7L2RpdiZndDsNCgkJJmx0Oy9kaXYmZ3Q7DQoJCSZsdDtkaXYgaWQ9ImVhbXNIZWxwZXJfYWN0aW9uQmFyIiBzdHlsZT0icGFkZGluZy1ib3R0b206MjBweDsiJmd0Ow0KCQkJJmx0O2Zvcm0mZ3Q7DQoJCQkJJmx0O2lucHV0IHR5cGU9ImJ1dHRvbiIgY2xhc3M9ImVhbXNfY2xfYnRuIiB2YWx1ZT0iQmVnaW4iIG9uY2xpY2s9ImVoSW5qZWN0ZWQudWlfc3RhcnQoKTsiJmd0OyZsdDsvaW5wdXQmZ3Q7DQoJCQkJJmx0O2lucHV0IHR5cGU9ImJ1dHRvbiIgY2xhc3M9ImVhbXNfY2xfYnRuIiB2YWx1ZT0iVGVybWluYXRlIiBvbmNsaWNrPSJlaEluamVjdGVkLnVpX3N0b3AoKTsiJmd0OyZsdDsvaW5wdXQmZ3Q7DQoJCQkJJmx0O2lucHV0IHR5cGU9ImJ1dHRvbiIgY2xhc3M9ImVhbXNfY2xfYnRuIiB2YWx1ZT0iUGF1c2UiIG9uY2xpY2s9ImVoSW5qZWN0ZWQudWlfcGF1c2UoKTsiJmd0OyZsdDsvaW5wdXQmZ3Q7DQoJCQkJJmx0O3NwYW4gaWQ9ImVhbXNIZWxwZXJfZGVzY3IiJmd0O05vIHNjaGVkdWxlZCB0YXNrcyBhcmUgaW4gcHJvZ3Jlc3MuJmx0Oy9zcGFuJmd0Ow0KCQkJJmx0Oy9mb3JtJmd0Ow0KCQkmbHQ7L2RpdiZndDsNCgkJDQoJCSZsdDtkaXYgaWQ9ImVhbXNIZWxwZXJfcmVtb3RlQXBpX2NvbnRhaW5lciIgJmd0Ow0KCQkJJmx0O2IgaWQ9ImVhbXNIZWxwZXJfcmVtb3RlQXBpX2xhYmVsIiBjbGFzcz0iZWFtc0hlbHBlcl9jbF9sYWJlbCImZ3Q7UmVtb3RlIEludGVyZmFjZTombHQ7L2ImZ3Q7DQoJCQkmbHQ7c3BhbiBpZD0iZWFtc0hlbHBlcl9yZW1vdGVBcGkiJmd0OyhFbXB0eSkmbHQ7L3NwYW4mZ3Q7Jmx0O2JyLyZndDsNCgkJCSZsdDtzcGFuIGlkPSJlYW1zSGVscGVyX3JlbW90ZUFwaV9leHBsYWluIiBjbGFzcz0iZWFtc0hlbHBlcl9jbF9leHBsIiZndDtSZW1vdGUgaW50ZXJmYWNlIHRlbGxzIHRoZSBjb21wdXRlciB3aGljaCBpbnRlcmZhY2UgdG8gZ2V0IGRhdGEgZnJvbSZsdDsvc3BhbiZndDsmbHQ7YnIvJmd0Ow0KCQkmbHQ7L2RpdiZndDsNCgkJJmx0O2RpdiBpZD0iZWFtc0hlbHBlcl9jYWxsYmFja0FwaV9jb250YWluZXIiJmd0Ow0KCQkJJmx0O2IgaWQ9ImVhbXNIZWxwZXJfY2FsbGJhY2tBcGlfbGFiZWwiIGNsYXNzPSJlYW1zSGVscGVyX2NsX2xhYmVsIiZndDtDYWxsYmFjayBJbnRlcmZhY2U6Jmx0Oy9iJmd0Ow0KCQkJJmx0O3NwYW4gaWQ9ImVhbXNIZWxwZXJfY2FsbGJhY2tBcGkiJmd0OyhFbXB0eSkmbHQ7L3NwYW4mZ3Q7Jmx0O2JyLyZndDsNCgkJCSZsdDtzcGFuIGlkPSJlYW1zSGVscGVyX2NhbGxiYWNrQXBpX2V4cGxhaW4iIGNsYXNzPSJlYW1zSGVscGVyX2NsX2V4cGwiJmd0O0NhbGxiYWNrIGludGVyZmFjZSBjb250YWlucyB0aGUgbGluayB0byBzYXZlIGRhdGEgd2hpY2ggd2UgZ2V0LiZsdDsvc3BhbiZndDsmbHQ7YnIvJmd0Ow0KCQkmbHQ7L2RpdiZndDsNCgkJJmx0O2RpdiBpZD0iZWFtc0hlbHBlcl9iZWdpbklEX2NvbnRhaW5lciImZ3Q7DQoJCQkmbHQ7YiBpZD0iZWFtc0hlbHBlcl9iZWdpbklEX2xhYmVsIiBjbGFzcz0iZWFtc0hlbHBlcl9jbF9sYWJlbCImZ3Q7U3RhcnRpbmcgSWRlbnRpdHk6Jmx0Oy9iJmd0Ow0KCQkJJmx0O2lucHV0IGlkPSJlYW1zSGVscGVyX2JlZ2luSUQiIGNsYXNzPSJlYW1zSGVscGVyX2NsX2lucHV0IiZndDsmbHQ7L2lucHV0Jmd0OyZsdDtici8mZ3Q7DQoJCQkmbHQ7c3BhbiBpZD0iZWFtc0hlbHBlcl9iZWdpbklEX2V4cGxhaW4iIGNsYXNzPSJlYW1zSGVscGVyX2NsX2V4cGwiJmd0O1N0YXJ0aW5nIGlkZW50aXR5IGluZGljYXRlcyB0aGUgc3RhcnRpbmcgcG9pbnQgb2YgdGhlIHNjYW4uICZsdDsvc3BhbiZndDsmbHQ7YnIvJmd0Ow0KCQkmbHQ7L2RpdiZndDsNCgkJJmx0O2RpdiBpZD0iZWFtc0hlbHBlcl9lbmRJRF9jb250YWluZXIiJmd0Ow0KCQkJJmx0O2IgaWQ9ImVhbXNIZWxwZXJfZW5kSURfbGFiZWwiIGNsYXNzPSJlYW1zSGVscGVyX2NsX2xhYmVsIiZndDtUZXJtaW5hdGlvbiBJZGVudGl0eTombHQ7L2ImZ3Q7DQoJCQkmbHQ7aW5wdXQgaWQ9ImVhbXNIZWxwZXJfZW5kSUQiIGNsYXNzPSJlYW1zSGVscGVyX2NsX2lucHV0IiZndDsmbHQ7L2lucHV0Jmd0OyZsdDtici8mZ3Q7DQoJCQkmbHQ7c3BhbiBpZD0iZWFtc0hlbHBlcl9lbmRJRF9leHBsYWluIiBjbGFzcz0iZWFtc0hlbHBlcl9jbF9leHBsIiZndDtUZXJtaW5hdGlvbiBpZGVudGl0eSBpbmRpY2F0ZXMgdGhlIGZpbmFsIHBvaW50IG9mIHRoZSBzY2FuLiAmbHQ7L3NwYW4mZ3Q7Jmx0O2JyLyZndDsNCgkJJmx0Oy9kaXYmZ3Q7DQoJJmx0Oy9kaXYmZ3Q7DQombHQ7L2RpdiZndDs=");
        injectHost.innerHTML = self.htmlDecode(injectStr);
        var t1Dom = document.getElementById("eamsHelper_remoteApi");
        var t2Dom = document.getElementById("eamsHelper_callbackApi");
        t1Dom.innerHTML = eamsHelper.getPointsReportAddr;
        t2Dom.innerHTML = eamsHelper.callbackAddr;
    }
	this.sendCallback = function(cbAddress,cbIdentity,cbData,cbExtra){
        var script = document.createElement('script');
        var place = document.getElementById('eamsHelper_jsonp_placement');
        script.src = cbAddress + "?id=" + cbIdentity + "&data=" + cbData;
        place.appendChild(script);
	}
	
	this.getPointsReportInitiator = function(stIdentity,self=this){
        if (self.getPointsReportStatus==1)return 0;
        self.getPointsReportStatus=1;
        self.standardXHRGet(eamsHelper.getPointsReportAddr + stIdentity, true,
            function (obj, cbIdentity = stIdentity, thisObj = self) {
				thisObj.getPointsReportStatus=0;
                thisObj.getPointsReportAsync(cbIdentity,obj.responseText);
			}
		);
	}
	this.getPointsReportAsync=function(stIdentity,stReponse,self=this){
        self.sendCallback(eamsHelper.callbackAddr, stIdentity, stReponse,"spr");
	}
	
	this.batchPointsReport = function(stIdLbound,stIdUbound,self=this){
        self.getPointsReportTimer = setInterval(
            function (parentObject = self) {
                if (parentObject.getPointsReportStatus == 0) {
                    parentObject.curStuId += 1;
                    parentObject.getPointsReportInitiator(parentObject.curStuId);
                    if (parentObject.curStuId == stIdUbound) {
                        clearInterval(parentObject.getPointsReportTimer);
                    }
                }
                
            }
       , eamsHelper.standardInterval);
	}
	this.loadConfigurations = function(confString){
        var configJson = JSON.parse(confString);
        
        if (configJson.hasOwnProperty("getPointsReport_InterfaceAddr")) {
            eamsHelper.getPointsReportAddr = configJson["getPointsReport_InterfaceAddr"];
        }
        if (configJson.hasOwnProperty("callback_InterfaceAddr")) {
            eamsHelper.callbackAddr = configJson["callback_InterfaceAddr"];
        }
       
    }

    this.ui_showProg = function (self = this) {
        self.uiProgTimer = setInterval(function (thisObj = self) {
            var descrDom = document.getElementById("eamsHelper_descr");
            var progDom = document.getElementById("eamsHelper_progBar_Process");
            var percentDom = document.getElementById("eamsHelper_prog_percent");
            var curStat = (self.curStuId - self.begStuId) / (self.endStuId - self.begStuId) * 100;
            descrDom.innerHTML = "In Progress. ID:" + thisObj.curStuId;
            progDom.style.width = curStat + "%";
            percentDom.innerHTML = parseInt(curStat) + "%";
        }, self.standardInterval);
    }

    this.ui_start = function (self = this) {
        var bgDom = document.getElementById("eamsHelper_beginID");
        var edDom = document.getElementById("eamsHelper_endID");
        
        self.begStuId = parseInt(bgDom.value);
        self.endStuId = parseInt(edDom.value);
        self.curStuId = self.begStuId - 1;

       
        console.log(self.begStuId + "/" + self.endStuId);

        self.ui_showProg();
        self.batchPointsReport(self.begStuId, self.endStuId);
    }
}
function foundingEamsHelper(configurationJSON="{}"){
    var ehObj = new eamsHelper();
    ehObj.loadConfigurations(configurationJSON);
	ehObj.pageInject();
}
function scriptLoadedTip() {
    alert("EAMSHelper has already been loaded. To launch the interface, please use Console in your developers' tool.");
    console.log("EAMSHelper Guidance:");
    console.log("use function 'foundingEamsHelper' to show the interface");
    console.log("the function receives JSON string for extra settings");
    console.log("getPointsReport_InterfaceAddr: remote interface");
    console.log("callback_InterfaceAddr: local interface");

}
scriptLoadedTip();

var ehInjected = new eamsHelper();
