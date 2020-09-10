// ==UserScript==
// @name         EAMSHelper
// @namespace    https://github.com/Aeroraven/Hello_World/EAMSHelper
// @version      0.3
// @description  Archive information on EAM System.
// @author       Aeroraven
// @match        *
// @grant        none
// @include      *
// ==/UserScript==

//End of Tampermonkey header

function eamsHelper() {
	eamsHelper.getPointsReportAddr="(Empty)";
	eamsHelper.getPointsReportCacheAddr="";
    eamsHelper.standardInterval = 500;
    eamsHelper.standardSliceLength = 800;
    eamsHelper.standardFileSliceLength = 5000;
    eamsHelper.callbackAddr = "(Empty)";
    eamsHelper.callbackType = 0;//0:URL / 1:FileDownload
    eamsHelper.argumentsEncodeType = 0;//0:Original / 1:b64
    eamsHelper.customizedXHRHeader = [];

	this._this=this;
	this.getPointsReportStatus=0;
    this.getPointsReportTimer = 0;

    this.fileContentArray = 0;
    this.fileContentArrayCur = 0;
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
    this.standardFileSave = function (sFileName, sContent) {
        var virtualDom = document.createElement("a");
        virtualDom.setAttribute("href", "data:text/plain;charset=utf-8," + encodeURIComponent(sContent));
        virtualDom.setAttribute("download", sFileName);

        console.log("SAVE:" + "data:text/plain;charset=utf-8," + encodeURIComponent(sContent));
        if (virtualDom.createEvent) {
            var ev = virtualDom.createEvent("MouseEvents");
            ev.initEvent("click", true, true);
            virtualDom.dispatchEvent(ev);
        } else {
            virtualDom.click();
        }
    }
	this.standardXHRGet = function(sUrl,bAsync,arrHeader,fOnSuccess=function(obj){},fOnError=function(errId){}){
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
        obj.open("GET", sUrl, bAsync);
        arrHeader.forEach(function (item, index) {
            if (item.hasOwnProperty("specialValue")) {
                if (item.specialValue.type = "localStorage.getItem") {
                    obj.setRequestHeader(item.name, localStorage.getItem(item.specialValue.name));
                }
            } else {
                obj.setRequestHeader(item.name, item.value);
            }
            
        });
		
		obj.send();
	}
    this.pageInject = function (self=this) {
        var injectHost = document.getElementsByTagName("body")[0];
        var injectStr = atob("Jmx0O3NjcmlwdCBzcmM9ImVhbXNIZWxwZXIuanMiJmd0OyZsdDsvc2NyaXB0Jmd0Ow0KDQombHQ7ZGl2IGlkPSJlYW1zSGVscGVyX2luamVjdG9yX21haW4iIHN0eWxlPSJiYWNrZ3JvdW5kLWNvbG9yOiMwMDAwMDA7Y29sb3I6I2ZmZmZmZjtsZWZ0OjBweDtyaWdodDowcHg7dG9wOjBweDtib3R0b206MHB4O3Bvc2l0aW9uOmZpeGVkO21hcmdpbi1sZWZ0OjBweDttYXJnaW4tcmlnaHQ6MHB4O21hcmdpbi10b3A6MHB4O21hcmdpbi1ib3R0b206MHB4O2ZvbnQtc2l6ZToxN3B4OyImZ3Q7DQoJJmx0O3NjcmlwdCZndDsNCgkJLy9FQU1TSGVscGVyIENhbGxlcg0KCQkNCgkmbHQ7L3NjcmlwdCZndDsNCgkmbHQ7ZGl2IGlkPSJlYW1zSGVscGVyX2pzb25wX3BsYWNlbWVudCImZ3Q7DQoJJmx0Oy9kaXYmZ3Q7DQoJJmx0O3N0eWxlJmd0Ow0KCS5lYW1zSGVscGVyX2NsX2V4cGx7DQoJCWNvbG9yOiNhYWFhYWE7DQoJCWZvbnQtc2l6ZToxNHB4Ow0KCQlwYWRkaW5nLWxlZnQ6MTBweDsNCgkJbWFyZ2luLXRvcDo4cHg7DQoJCWRpc3BsYXk6YmxvY2s7DQoJfQ0KCS5lYW1zX2NsX2J0bnsNCgkJZm9udC1zaXplOjE4cHg7DQoJCWJvcmRlci1yYWRpdXM6NHB4Ow0KCQliYWNrZ3JvdW5kLWNvbG9yOiM0MDhkZWE7DQoJCWNvbG9yOiNmZmZmZmY7DQoJCWJvcmRlcjowcHg7DQoJCXBhZGRpbmctdG9wOjEwcHg7DQoJCXBhZGRpbmctYm90dG9tOjEwcHg7DQoJCXBhZGRpbmctbGVmdDoyMHB4Ow0KCQlwYWRkaW5nLXJpZ2h0OjIwcHg7DQoJCW1hcmdpbi1yaWdodDo1cHg7DQoJfQ0KCS5lYW1zSGVscGVyX2NsX2lucHV0ew0KCQlib3JkZXItdG9wOjBweDsNCgkJYm9yZGVyLWxlZnQ6MHB4Ow0KCQlib3JkZXItcmlnaHQ6MHB4Ow0KCQliYWNrZ3JvdW5kLWNvbG9yOiMwMDAwMDA7DQoJCWJvcmRlci1ib3R0b206MnB4ICMyMDZkY2Egc29saWQ7DQoJCWNvbG9yOiNmZmZmZmY7DQoJCWZvbnQtc2l6ZToyMHB4Ow0KCX0NCgkuZWFtc0hlbHBlcl9jbF9sYWJlbHsNCgkJY29sb3I6IzQwOGRlYTsNCgl9DQoJJmx0Oy9zdHlsZSZndDsNCgkmbHQ7ZGl2IGlkPSJlYW1zSGVscGVyX3RpdGxlIiBzdHlsZT0iYmFja2dyb3VuZC1jb2xvcjojMjA2ZGNhO3BhZGRpbmctdG9wOjI1cHg7cGFkZGluZy1ib3R0b206MjVweDtwYWRkaW5nLWxlZnQ6MTVweDtmb250LXNpemU6MjVweDtmb250LWZhbWlseTpiYWhuc2NocmlmdDsiJmd0O0VBTVNIZWxwZXImbHQ7L2RpdiZndDsNCgkmbHQ7ZGl2IGlkPSJlYW1zSGVscGVyX21haW4iIHN0eWxlPSJwYWRkaW5nLWxlZnQ6NDBweDtwYWRkaW5nLXRvcDoyNXB4IiZndDsNCgkJJmx0O2RpdiBpZD0iZWFtc0hlbHBlcl9wcm9nQmFyIiZndDsNCgkJCSZsdDtkaXYgc3R5bGU9Im1hcmdpbjowIGF1dG87IiZndDsNCgkJCQkmbHQ7dGFibGUgc3R5bGU9IndpZHRoOjkwJSImZ3Q7DQoJCQkJCSZsdDt0ciBzdHlsZT0id2lkdGg6OTAlIiZndDsNCgkJCQkJCSZsdDt0ZCBzdHlsZT0id2lkdGg6ODAlIiZndDsNCgkJCQkJCQkmbHQ7ZGl2IGlkPSJlYW1zSGVscGVyX3Byb2dCYXJfTWFpbiIgc3R5bGU9Im1hcmdpbi10b3A6MTBweDttYXJnaW4tYm90dG9tOjEwcHg7d2lkdGg6MTAwJTtiYWNrZ3JvdW5kLWNvbG9yOiNhYWFhYWE7aGVpZ2h0OjQwcHg7ZGlzcGxheTppbmxpbmUtYmxvY2s7Ym9yZGVyLXJhZGl1czo1cHg7IiZndDsNCgkJCQkJCQkJJmx0O2RpdiBpZD0iZWFtc0hlbHBlcl9wcm9nQmFyX1Byb2Nlc3MiIHN0eWxlPSJ0ZXh0LWFsaWduOmxlZnQ7bWFyZ2luLWJvdHRvbToxMHB4O3dpZHRoOjElO2JhY2tncm91bmQtY29sb3I6IzIwNmRjYTtoZWlnaHQ6NDBweDtkaXNwbGF5OmlubGluZS1ibG9jaztib3JkZXItcmFkaXVzOjVweDsiJmd0Ow0KCQkJCQkJCQ0KCQkJCQkJCQkmbHQ7L2RpdiZndDsNCgkJCQkJCQkmbHQ7L2RpdiZndDsNCgkJCQkJCSZsdDsvdGQmZ3Q7DQoJCQkJCQkmbHQ7dGQgc3R5bGU9IndpZHRoOjUlIiZndDsNCgkJCQkJCQkmbHQ7c3BhbiBpZD0iZWFtc0hlbHBlcl9wcm9nX3BlcmNlbnQiIHN0eWxlPSJjb2xvcjojZmZmZmZmO2ZvbnQtd2VpZ2h0OmJvbGQ7Zm9udC1zaXplOjI1cHg7cGFkZGluZy1sZWZ0OjVweDsiJmd0OzAlJmx0Oy9zcGFuJmd0Ow0KCQkJCQkJJmx0Oy90ZCZndDsNCgkJCQkJJmx0Oy90ciZndDsNCgkJCQkmbHQ7L3RhYmxlJmd0Ow0KCQkJCQ0KCQkJJmx0Oy9kaXYmZ3Q7DQoJCSZsdDsvZGl2Jmd0Ow0KCQkmbHQ7ZGl2IGlkPSJlYW1zSGVscGVyX2FjdGlvbkJhciIgc3R5bGU9InBhZGRpbmctYm90dG9tOjIwcHg7IiZndDsNCgkJCSZsdDtmb3JtJmd0Ow0KCQkJCSZsdDtpbnB1dCB0eXBlPSJidXR0b24iIGNsYXNzPSJlYW1zX2NsX2J0biIgdmFsdWU9IkJlZ2luIiBvbmNsaWNrPSJ3aW5kb3cuZWhJbmplY3RlZC51aV9zdGFydCgpOyImZ3Q7Jmx0Oy9pbnB1dCZndDsNCgkJCQkmbHQ7aW5wdXQgdHlwZT0iYnV0dG9uIiBjbGFzcz0iZWFtc19jbF9idG4iIHZhbHVlPSJUZXJtaW5hdGUiIG9uY2xpY2s9IndpbmRvdy5laEluamVjdGVkLnVpX3N0b3AoKTsiJmd0OyZsdDsvaW5wdXQmZ3Q7DQoJCQkJJmx0O2lucHV0IHR5cGU9ImJ1dHRvbiIgY2xhc3M9ImVhbXNfY2xfYnRuIiB2YWx1ZT0iUGF1c2UiIG9uY2xpY2s9IndpbmRvdy5laEluamVjdGVkLnVpX3BhdXNlKCk7IiZndDsmbHQ7L2lucHV0Jmd0Ow0KCQkJCSZsdDtzcGFuIGlkPSJlYW1zSGVscGVyX2Rlc2NyIiZndDtObyBzY2hlZHVsZWQgdGFza3MgYXJlIGluIHByb2dyZXNzLiZsdDsvc3BhbiZndDsNCgkJCSZsdDsvZm9ybSZndDsNCgkJJmx0Oy9kaXYmZ3Q7DQoJCQ0KCQkmbHQ7ZGl2IGlkPSJlYW1zSGVscGVyX3JlbW90ZUFwaV9jb250YWluZXIiICZndDsNCgkJCSZsdDtiIGlkPSJlYW1zSGVscGVyX3JlbW90ZUFwaV9sYWJlbCIgY2xhc3M9ImVhbXNIZWxwZXJfY2xfbGFiZWwiJmd0O1JlbW90ZSBJbnRlcmZhY2U6Jmx0Oy9iJmd0Ow0KCQkJJmx0O3NwYW4gaWQ9ImVhbXNIZWxwZXJfcmVtb3RlQXBpIiZndDsoRW1wdHkpJmx0Oy9zcGFuJmd0OyZsdDtici8mZ3Q7DQoJCQkmbHQ7c3BhbiBpZD0iZWFtc0hlbHBlcl9yZW1vdGVBcGlfZXhwbGFpbiIgY2xhc3M9ImVhbXNIZWxwZXJfY2xfZXhwbCImZ3Q7UmVtb3RlIGludGVyZmFjZSB0ZWxscyB0aGUgY29tcHV0ZXIgd2hpY2ggaW50ZXJmYWNlIHRvIGdldCBkYXRhIGZyb20mbHQ7L3NwYW4mZ3Q7Jmx0O2JyLyZndDsNCgkJJmx0Oy9kaXYmZ3Q7DQoJCSZsdDtkaXYgaWQ9ImVhbXNIZWxwZXJfY2FsbGJhY2tBcGlfY29udGFpbmVyIiZndDsNCgkJCSZsdDtiIGlkPSJlYW1zSGVscGVyX2NhbGxiYWNrQXBpX2xhYmVsIiBjbGFzcz0iZWFtc0hlbHBlcl9jbF9sYWJlbCImZ3Q7Q2FsbGJhY2sgSW50ZXJmYWNlOiZsdDsvYiZndDsNCgkJCSZsdDtzcGFuIGlkPSJlYW1zSGVscGVyX2NhbGxiYWNrQXBpIiZndDsoRW1wdHkpJmx0Oy9zcGFuJmd0OyZsdDtici8mZ3Q7DQoJCQkmbHQ7c3BhbiBpZD0iZWFtc0hlbHBlcl9jYWxsYmFja0FwaV9leHBsYWluIiBjbGFzcz0iZWFtc0hlbHBlcl9jbF9leHBsIiZndDtDYWxsYmFjayBpbnRlcmZhY2UgY29udGFpbnMgdGhlIGxpbmsgdG8gc2F2ZSBkYXRhIHdoaWNoIHdlIGdldC4mbHQ7L3NwYW4mZ3Q7Jmx0O2JyLyZndDsNCgkJJmx0Oy9kaXYmZ3Q7DQoJCSZsdDtkaXYgaWQ9ImVhbXNIZWxwZXJfYmVnaW5JRF9jb250YWluZXIiJmd0Ow0KCQkJJmx0O2IgaWQ9ImVhbXNIZWxwZXJfYmVnaW5JRF9sYWJlbCIgY2xhc3M9ImVhbXNIZWxwZXJfY2xfbGFiZWwiJmd0O1N0YXJ0aW5nIElkZW50aXR5OiZsdDsvYiZndDsNCgkJCSZsdDtpbnB1dCBpZD0iZWFtc0hlbHBlcl9iZWdpbklEIiBjbGFzcz0iZWFtc0hlbHBlcl9jbF9pbnB1dCImZ3Q7Jmx0Oy9pbnB1dCZndDsmbHQ7YnIvJmd0Ow0KCQkJJmx0O3NwYW4gaWQ9ImVhbXNIZWxwZXJfYmVnaW5JRF9leHBsYWluIiBjbGFzcz0iZWFtc0hlbHBlcl9jbF9leHBsIiZndDtTdGFydGluZyBpZGVudGl0eSBpbmRpY2F0ZXMgdGhlIHN0YXJ0aW5nIHBvaW50IG9mIHRoZSBzY2FuLiAmbHQ7L3NwYW4mZ3Q7Jmx0O2JyLyZndDsNCgkJJmx0Oy9kaXYmZ3Q7DQoJCSZsdDtkaXYgaWQ9ImVhbXNIZWxwZXJfZW5kSURfY29udGFpbmVyIiZndDsNCgkJCSZsdDtiIGlkPSJlYW1zSGVscGVyX2VuZElEX2xhYmVsIiBjbGFzcz0iZWFtc0hlbHBlcl9jbF9sYWJlbCImZ3Q7VGVybWluYXRpb24gSWRlbnRpdHk6Jmx0Oy9iJmd0Ow0KCQkJJmx0O2lucHV0IGlkPSJlYW1zSGVscGVyX2VuZElEIiBjbGFzcz0iZWFtc0hlbHBlcl9jbF9pbnB1dCImZ3Q7Jmx0Oy9pbnB1dCZndDsmbHQ7YnIvJmd0Ow0KCQkJJmx0O3NwYW4gaWQ9ImVhbXNIZWxwZXJfZW5kSURfZXhwbGFpbiIgY2xhc3M9ImVhbXNIZWxwZXJfY2xfZXhwbCImZ3Q7VGVybWluYXRpb24gaWRlbnRpdHkgaW5kaWNhdGVzIHRoZSBmaW5hbCBwb2ludCBvZiB0aGUgc2Nhbi4gJmx0Oy9zcGFuJmd0OyZsdDtici8mZ3Q7DQoJCSZsdDsvZGl2Jmd0Ow0KCSZsdDsvZGl2Jmd0Ow0KJmx0Oy9kaXYmZ3Q7");
        injectHost.innerHTML = self.htmlDecode(injectStr);
        var t1Dom = document.getElementById("eamsHelper_remoteApi");
        var t2Dom = document.getElementById("eamsHelper_callbackApi");
        t1Dom.innerHTML = eamsHelper.getPointsReportAddr;
        if (eamsHelper.callbackType != 1) {
            t2Dom.innerHTML = eamsHelper.callbackAddr;
        } else {
            t2Dom.innerHTML = "(Save As File)"
        }
        

        var clearHost = document.getElementsByTagName("head")[0];
        clearHost.innerHTML = "";
    }
    this.sendCallback = function (cbAddress, cbIdentity, cbData, cbExtra, self=this) {
        if (eamsHelper.callbackType == 0) {
            var place = document.getElementById('eamsHelper_jsonp_placement');
            var cbLen = cbData.length;
            var cbSlices = parseInt(cbData.length / eamsHelper.standardSliceLength);
            var cbSlicedData = "";
            for (var i = 0; i <= cbSlices; i++) {
                if (i != cbSlices) {
                    cbSlicedData = cbData.substring(i * eamsHelper.standardSliceLength, (i + 1) * eamsHelper.standardSliceLength);
                } else {
                    cbSlicedData = cbData.substring(i * eamsHelper.standardSliceLength, cbLen);
                }
                var script = document.createElement('script');
                script.src = cbAddress + "?id=" + cbIdentity + "&data=" + cbSlicedData + "&slice=" + i;
                place.appendChild(script);
            }
        } else if (eamsHelper.callbackType == 1) {
            self.fileContentArrayCur++;
            self.fileContentArray[self.fileContentArrayCur] = {};
            self.fileContentArray[self.fileContentArrayCur]['id'] = cbIdentity;
            self.fileContentArray[self.fileContentArrayCur]['data'] = cbData;

            
        }
        
       
	}
    this.saveAsFile = function (self = this) {
        var saveContent = JSON.stringify(self.fileContentArray);
        var cbLen = saveContent.length;
        var cbSlices = parseInt(saveContent.length / eamsHelper.standardFileSliceLength);
        var cbSlicedData = "";
        var fileNamePrefix = prompt("Please set name for file to be saved:", "EAMSHelperSave_")

        for (var i = 0; i <= cbSlices; i++) {
            if (i != cbSlices) {
                cbSlicedData = saveContent.substring(i * eamsHelper.standardFileSliceLength, (i + 1) * eamsHelper.standardFileSliceLength);
            } else {
                cbSlicedData = saveContent.substring(i * eamsHelper.standardFileSliceLength, cbLen);
            }
            self.standardFileSave(fileNamePrefix + i+".txt", cbSlicedData);
        }
    }

	this.getPointsReportInitiator = function(stIdentity,self=this){
        if (self.getPointsReportStatus == 1) return 0;

        var xpIdentity;
        if (eamsHelper.argumentsEncodeType == 0) {
            xpIdentity = stIdentity;
        } else {
            xpIdentity = btoa(stIdentity);
        }

        self.getPointsReportStatus = 1;
        self.standardXHRGet(eamsHelper.getPointsReportAddr + xpIdentity, true, eamsHelper.customizedXHRHeader,
            function (obj, cbIdentity = stIdentity, thisObj = self) {
				thisObj.getPointsReportStatus=0;
                thisObj.getPointsReportAsync(cbIdentity,obj.responseText);
			}
		);
	}
    this.getPointsReportAsync = function (stIdentity, stReponse, self = this) {
        var sbData = btoa(unescape(encodeURIComponent(stReponse)));
        sbData = sbData.replace(/[+]/g, "$");
        self.sendCallback(eamsHelper.callbackAddr, stIdentity, sbData, "spr");
        if (eamsHelper.callbackType == 1 && self.curStuId == self.endStuId) {
            self.saveAsFile();
        }
        console.log(sbData);
	}
	
	this.batchPointsReport = function(stIdLbound,stIdUbound,self=this){
        self.getPointsReportTimer = setInterval(
            function (parentObject = self) {
                if (parentObject.getPointsReportStatus == 0) {
                    parentObject.curStuId += 1;
                    parentObject.getPointsReportInitiator(parentObject.curStuId);
                    if (parentObject.curStuId == stIdUbound) {
                        clearInterval(parentObject.getPointsReportTimer);
                        alert("Task Completed!");
                        
                    }
                }
                
            }
       , eamsHelper.standardInterval);
	}
	this.loadConfigurations = function(confString,isJson=true){
        var configJson;
        if (isJson == true) {
            configJson = configJson;
        } else {
            configJson = JSON.parse(confString);
        }
        
        if (configJson.hasOwnProperty("getPointsReport_InterfaceAddr")) {
            eamsHelper.getPointsReportAddr = configJson["getPointsReport_InterfaceAddr"];
        }
        if (configJson.hasOwnProperty("callback_InterfaceAddr")) {
            eamsHelper.callbackAddr = configJson["callback_InterfaceAddr"];
        }
        if (configJson.hasOwnProperty("callback_Type")) {
            eamsHelper.callbackType = configJson["callback_Type"];
        }
        if (configJson.hasOwnProperty("arguments_EncodingType")) {
            eamsHelper.argumentsEncodeType = configJson["arguments_EncodingType"];
        }
        if (configJson.hasOwnProperty("customized_Header")) {
            eamsHelper.customizedXHRHeader = configJson["customized_Header"];
        }
        if (configJson.hasOwnProperty("timer_Interval")) {
            eamsHelper.standardInterval = configJson["timer_Interval"];
        }
        if (configJson.hasOwnProperty("url_DataLength")) {
            eamsHelper.standardSliceLength = configJson["url_DataLength"];
        }
        if (configJson.hasOwnProperty("file_DataLength")) {
            eamsHelper.standardFileSliceLength = configJson["file_DataLength"];
        }
    }

    this.ui_showProg = function (self = this) {
        self.uiProgTimer = setInterval(function (thisObj = self) {
            var descrDom = document.getElementById("eamsHelper_descr");
            var progDom = document.getElementById("eamsHelper_progBar_Process");
            var percentDom = document.getElementById("eamsHelper_prog_percent");
            var curStat = (self.curStuId - self.begStuId+1) / (self.endStuId - self.begStuId+1) * 100;
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
        self.fileContentArray = {};
    }
    this.ui_prestart_json = function (self = this) {
        if (!confirm("Launching EAMSHelper will completely remove almost all elements in the page. The removal is unchangeable until you reload the page. Are you sure to launch EAMSHelper?")) {
            return;
        }
        var confstr = prompt("Input configuration JSON string below:", "{}");
        this.loadConfigurations(confstr, false);
        this.pageInject();
    }
    this.ui_prestart = function (self = this) {
        alert("Sorry, the function is not available now.")
    }

    this.preInject = function (self = this) {
        var injectHost = document.getElementsByTagName("body")[0];
        var injectStr = atob("Jmx0O2RpdiBpZD0iZWFtc0hlbHBlclByZWxvYWRlcl9pbmplY3Rvcl9tYWluIiBzdHlsZT0iYmFja2dyb3VuZC1jb2xvcjojMDAwMDAwO2NvbG9yOiNmZmZmZmY7bGVmdDowcHg7cmlnaHQ6MHB4O3RvcDowcHg7cG9zaXRpb246Zml4ZWQ7bWFyZ2luLWxlZnQ6MHB4O21hcmdpbi1yaWdodDowcHg7bWFyZ2luLXRvcDowcHg7bWFyZ2luLWJvdHRvbTowcHg7Zm9udC1zaXplOjE3cHg7IiZndDsNCgkmbHQ7c2NyaXB0Jmd0Ow0KCQkvL0VBTVNIZWxwZXIgQ2FsbGVyDQoJCQ0KCSZsdDsvc2NyaXB0Jmd0Ow0KCSZsdDtkaXYgaWQ9ImVhbXNIZWxwZXJQcmVsb2FkZXJfanNvbnBfcGxhY2VtZW50IiZndDsNCgkmbHQ7L2RpdiZndDsNCgkmbHQ7c3R5bGUmZ3Q7DQoJLmVhbXNIZWxwZXJfY2xfZXhwbHsNCgkJY29sb3I6I2FhYWFhYTsNCgkJZm9udC1zaXplOjE0cHg7DQoJCXBhZGRpbmctbGVmdDoxMHB4Ow0KCQltYXJnaW4tdG9wOjhweDsNCgkJZGlzcGxheTpibG9jazsNCgl9DQoJLmVhbXNfY2xfYnRuMnsNCgkJZm9udC1zaXplOjE0cHg7DQoJCWJvcmRlci1yYWRpdXM6NHB4Ow0KCQliYWNrZ3JvdW5kLWNvbG9yOiM0MDhkZWE7DQoJCWNvbG9yOiNmZmZmZmY7DQoJCWJvcmRlcjowcHg7DQoJCXBhZGRpbmctdG9wOjEwcHg7DQoJCXBhZGRpbmctYm90dG9tOjEwcHg7DQoJCXBhZGRpbmctbGVmdDoyMHB4Ow0KCQlwYWRkaW5nLXJpZ2h0OjIwcHg7DQoJCW1hcmdpbi1yaWdodDo1cHg7DQoJfQ0KCS5lYW1zSGVscGVyX2NsX2lucHV0ew0KCQlib3JkZXItdG9wOjBweDsNCgkJYm9yZGVyLWxlZnQ6MHB4Ow0KCQlib3JkZXItcmlnaHQ6MHB4Ow0KCQliYWNrZ3JvdW5kLWNvbG9yOiMwMDAwMDA7DQoJCWJvcmRlci1ib3R0b206MnB4ICMyMDZkY2Egc29saWQ7DQoJCWNvbG9yOiNmZmZmZmY7DQoJCWZvbnQtc2l6ZToyMHB4Ow0KCX0NCgkuZWFtc0hlbHBlcl9jbF9sYWJlbHsNCgkJY29sb3I6IzQwOGRlYTsNCgl9DQoJJmx0Oy9zdHlsZSZndDsNCgkmbHQ7ZGl2IGlkPSJlYW1zSGVscGVyUHJlbG9hZGVyX3RpdGxlIiBzdHlsZT0iYmFja2dyb3VuZC1jb2xvcjojMjIyMjIyO2NvbG9yOiM0MDhkZWE7cGFkZGluZy10b3A6MTVweDtwYWRkaW5nLWJvdHRvbToxNXB4O3BhZGRpbmctbGVmdDoxNXB4O2ZvbnQtc2l6ZToxNHB4O2ZvbnQtZmFtaWx5OmJhaG5zY2hyaWZ0O3otaW5kZXg6OTk5OTk7IiZndDtUbyBsYXVuY2ggRUFNU0hlbHBlciwgQ2xpY2sgaGVyZTogJm5ic3A7Jm5ic3A7DQoJCSZsdDtpbnB1dCBpZD0iZWFtc0hlbHBlclByZWxvYWRlcl9sYXVuY2giIHR5cGU9YnV0dG9uIGNsYXNzPSJlYW1zX2NsX2J0bjIiIHZhbHVlPSJMYXVuY2giIG9uY2xpY2s9IndpbmRvdy5laEluamVjdGVkLnVpX3ByZXN0YXJ0KCk7IiZndDsmbHQ7L2lucHV0Jmd0Ow0KCQkmbHQ7aW5wdXQgaWQ9ImVhbXNIZWxwZXJQcmVsb2FkZXJfbGF1bmNoIiB0eXBlPWJ1dHRvbiBjbGFzcz0iZWFtc19jbF9idG4yIiB2YWx1ZT0iRGVidWcgKEFkdmFuY2VkKSIgb25jbGljaz0id2luZG93LmVoSW5qZWN0ZWQudWlfcHJlc3RhcnRfanNvbigpOyImZ3Q7Jmx0Oy9pbnB1dCZndDsNCgkmbHQ7L2RpdiZndDsNCgkNCiZsdDsvZGl2Jmd0Ow==");
        var injectContainer = document.createElement('div');
        injectContainer.innerHTML = self.htmlDecode(injectStr);
        
        injectContainer.style.zIndex = 99999;
        injectContainer.style.position = "absolute";
        injectHost.appendChild(injectContainer);
    }
}
function foundingEamsHelper(configurationJSON = {}){
    var ehObj = new eamsHelper();
    ehObj.loadConfigurations(configurationJSON);
	ehObj.pageInject();
}
function ehPreload() {
    var ehObj = new eamsHelper();
    ehObj.preInject();

}
function scriptLoadedTip() {
    console.log("EAMSHelper Loaded!");
}
scriptLoadedTip();
window.ehInjected = new eamsHelper();
ehPreload();
