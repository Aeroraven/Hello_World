!function(c){function t(t){for(var e,i,n=t[0],o=t[1],r=t[2],s=0,a=[];s<n.length;s++)i=n[s],Object.prototype.hasOwnProperty.call(u,i)&&u[i]&&a.push(u[i][0]),u[i]=0;for(e in o)Object.prototype.hasOwnProperty.call(o,e)&&(c[e]=o[e]);for(p&&p(t);a.length;)a.shift()();return m.push.apply(m,r||[]),l()}function l(){for(var t,e=0;e<m.length;e++){for(var i=m[e],n=!0,o=1;o<i.length;o++){var r=i[o];0!==u[r]&&(n=!1)}n&&(m.splice(e--,1),t=s(s.s=i[0]))}return t}var i={},u={app:0},m=[];function s(t){if(i[t])return i[t].exports;var e=i[t]={i:t,l:!1,exports:{}};return c[t].call(e.exports,e,e.exports,s),e.l=!0,e.exports}s.m=c,s.c=i,s.d=function(t,e,i){s.o(t,e)||Object.defineProperty(t,e,{enumerable:!0,get:i})},s.r=function(t){"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(t,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(t,"__esModule",{value:!0})},s.t=function(e,t){if(1&t&&(e=s(e)),8&t)return e;if(4&t&&"object"==typeof e&&e&&e.__esModule)return e;var i=Object.create(null);if(s.r(i),Object.defineProperty(i,"default",{enumerable:!0,value:e}),2&t&&"string"!=typeof e)for(var n in e)s.d(i,n,function(t){return e[t]}.bind(null,n));return i},s.n=function(t){var e=t&&t.__esModule?function(){return t.default}:function(){return t};return s.d(e,"a",e),e},s.o=function(t,e){return Object.prototype.hasOwnProperty.call(t,e)},s.p="app://./";var e=(n=window.webpackJsonp=window.webpackJsonp||[]).push.bind(n);n.push=t;for(var n=n.slice(),o=0;o<n.length;o++)t(n[o]);var p=e;m.push([0,"chunk-vendors"]),l()}({0:function(t,e,i){t.exports=i("56d7")},"0128":function(t,e,i){"use strict";i("2f98")},"034f":function(t,e,i){"use strict";i("85ec")},"05cd":function(t,e,i){"use strict";i("4a80")},"0958":function(t,e,i){"use strict";i("53f4")},"0d0d":function(t,e,i){"use strict";i("692b")},"2e90":function(t,e,i){"use strict";i("92ea")},"2f98":function(t,e,i){},"4a80":function(t,e,i){},"53f4":function(t,e,i){},"56d7":function(t,e,i){"use strict";i.r(e);var n=i("2b0e"),o={name:"TopBar",props:{title:String}},r=(i("89ff"),i("2877")),s=Object(r.a)(o,function(){var t=this.$createElement,t=this._self._c||t;return t("div",{staticClass:"oshw2_topbar"},[t("span",{domProps:{innerHTML:this._s(this.title)}})])},[],!1,null,"14df66e2",null).exports,a={name:"MemoryContainerElement",props:{eleAttrs:Object},created:function(){this.updateWidth()},watch:{eleAttrs:function(){this.updateWidth()}},methods:{updateWidth:function(){this.styAttr.width=(this.eleAttrs.el.end-this.eleAttrs.el.begin)/this.eleAttrs.global_data.maxMemory*100+"%",this.styAttr.left=this.eleAttrs.el.begin/this.eleAttrs.global_data.maxMemory*100+"%"},tacklingHover:function(t){this.hoverStat=t}},data:function(){return{styAttr:{width:"22px",height:"100%",left:"0px",top:"0px"},hoverStat:!1}},computed:{}},c=(i("05cd"),Object(r.a)(a,function(){var e=this,t=e.$createElement,t=e._self._c||t;return t("div",{staticClass:"oshw2_dynamic_mem_alloc_child",class:"oshw2_dynamic_mem_alloc_childx_"+e.eleAttrs.el.orderId%2,style:e.styAttr,on:{mouseover:function(t){return e.tacklingHover(!0)},mouseleave:function(t){return e.tacklingHover(!1)}}},[t("div",{staticClass:"oshw2_dynamic_mem_alloc_child_detailhw"},[t("b",[e._v("#")]),t("span",{domProps:{innerHTML:e._s(e.eleAttrs.el.jobId)}}),t("br")]),t("transition",{attrs:{name:"fade"}},[t("div",{directives:[{name:"show",rawName:"v-show",value:e.hoverStat,expression:"hoverStat"}],key:e.eleAttrs.id,staticClass:"oshw2_dynamic_mem_alloc_child_detail"},[t("div",{staticClass:"oshw2_dynamic_mem_alloc_child_detail"},[t("b",[e._v("起始内存地址:")]),t("span",{domProps:{innerHTML:e._s(e.eleAttrs.el.begin)}}),e._v(" B"),t("br"),t("b",[e._v("终止内存地址:")]),t("span",{domProps:{innerHTML:e._s(e.eleAttrs.el.end)}}),e._v(" B"),t("br"),t("b",[e._v("内存占用长度:")]),t("span",{domProps:{innerHTML:e._s(e.eleAttrs.el.end-e.eleAttrs.el.begin)}}),e._v(" B"),t("br")])])])],1)},[],!1,null,null,null)).exports,l={name:"MemoryContainer",props:{memContainerProp:Object,memWatch:Boolean,memInfoArg:Array},components:{MemoryContainerElement:c},watch:{memWatch:function(){this.MemoryData=this.memInfoArg}},data:function(){return{MemoryData:[]}}},e=(i("0d0d"),Object(r.a)(l,function(){var e=this,t=e.$createElement,i=e._self._c||t;return i("div",{staticClass:"oshw2_memory_container"},[i("transition-group",{attrs:{name:"fade2"}},e._l(e.MemoryData.length,function(t){return i("memory-container-element",{key:e.MemoryData[t-1].jobId,attrs:{eleAttrs:{el:e.MemoryData[t-1],global_data:e.memContainerProp,id:t}}})}),1)],1)},[],!1,null,"97f0a344",null)).exports,o={name:"CustButton",props:{caption:String,selectedProp:Boolean}},a=(i("2e90"),Object(r.a)(o,function(){var e=this,t=e.$createElement,t=e._self._c||t;return t("div",{staticClass:"oshw2_cust_button",class:"oshw2_cust_button_s"+e.selectedProp,on:{click:function(t){return e.$emit("cust_click")},hover:function(t){return e.$emit("cust_hover")}}},[t("span",{domProps:{innerHTML:e._s(e.caption)}})])},[],!1,null,"3e0cb852",null)).exports,c={name:"AllocReqTable",props:{allocReqTableProp:Array},data:function(){return{reqList:[]}},created:function(){this.reqList=this.allocReqTableProp}},l=(i("0958"),Object(r.a)(c,function(){var e=this,t=e.$createElement,i=e._self._c||t;return i("div",{staticClass:"oshw2_alloc_req_table"},[i("div",{staticClass:"oshw2_alloc_req_table_tbl"},[i("div",{staticClass:"oshw2_alloc_req_table_tbl_head oshw2_alloc_req_table_e"},[e._v("请求作业ID")]),i("div",{staticClass:"oshw2_alloc_req_table_tbl_head oshw2_alloc_req_table_e"},[e._v("请求操作")]),i("div",{staticClass:"oshw2_alloc_req_table_tbl_head oshw2_alloc_req_table_e"},[e._v("请求大小")]),i("div",{staticClass:"oshw2_alloc_req_table_tbl_slide"},[i("transition-group",{attrs:{name:"fade2"}},e._l(e.reqList.length,function(t){return i("div",{key:e.reqList[t-1].uniqId,staticClass:"oshw2_alloc_req_table_tbl_tr"},[i("div",{staticClass:"oshw2_alloc_req_table_e2 oshw2_alloc_req_table_e"},[e._v(e._s(e.reqList[t-1].jobId))]),i("div",{staticClass:"oshw2_alloc_req_table_e2 oshw2_alloc_req_table_e"},[e._v(e._s(e.reqList[t-1].reqType?"Request":"Release"))]),i("div",{staticClass:"oshw2_alloc_req_table_e2 oshw2_alloc_req_table_e"},[e._v(e._s(e.reqList[t-1].reqSize)+" B")]),i("br")])}),0)],1)])])},[],!1,null,"3b9d963e",null)).exports,o={name:"CustSelect",data(){return{activeIndex:0}},props:{dataList:{type:Array,default(){return[{name:"首次适应算法"},{name:"最佳适应算法"}]}},labelProperty:{type:String,default(){return"name"}}},directives:{dpl:{bind(t){t.style.display="none"}}},methods:{onDplOver(t){let e=t.currentTarget.childNodes[1];e.style.display="block"},onDplOut(t){let e=t.currentTarget.childNodes[1];e.style.display="none"},onLiClick(t){let e=event.path||event.composedPath&&event.composedPath();e[1].style.display="none",this.activeIndex=t,this.$emit("change",{index:t,value:this.dataList[t]})}},computed:{dplLable(){return this.dataList[this.activeIndex][this.labelProperty]}}},c=(i("b83d"),Object(r.a)(o,function(){var i=this,t=i.$createElement,n=i._self._c||t;return n("div",{staticClass:"oshw2_cust_select",on:{mouseover:function(t){return i.onDplOver(t)},mouseout:function(t){return i.onDplOut(t)}}},[n("span",{staticStyle:{position:"relative"}},[n("a",[i._v("▼")]),i._v(i._s(i.dplLable)),n("i")]),n("ul",{directives:[{name:"dpl",rawName:"v-dpl"}]},i._l(i.dataList,function(t,e){return n("li",{key:e,on:{click:function(t){i.onLiClick(e,t),i.$emit("cust_click",i.activeIndex)}}},[i._v(i._s(t[i.labelProperty]))])}),0)])},[],!1,null,"64b84840",null)).exports,o={name:"OperationRecord",props:{recordInfo:Object},watch:{recordInfo:function(){console.log(this.recordInfo.log)}},data:function(){return{reqList:[]}},created:function(){this.reqList=this.allocReqTableProp}},o=(i("0128"),Object(r.a)(o,function(){var t=this,e=t.$createElement,e=t._self._c||e;return e("div",{staticClass:"oshw2_oprecord"},[e("div",{staticClass:"oshw2_oprecord_tbl"},[e("div",{staticClass:"oshw2_oprecord_tbl_slide"},[e("pre",[t._v("          "+t._s(t.recordInfo.log)+"\n          ")])])])])},[],!1,null,"0909a37e",null)).exports,o={name:"DynamicMemAlloc",props:{dynamicMemAllocProp:Object},components:{MemoryContainer:e,CustButton:a,AllocReqTable:l,CustSelect:c,OperationRecord:o},data:function(){return{frameComponentList:["AllocReqTable","OperationRecord"],frameComponentActive:0,usingAlgo:0,dynamicWorkList:[],workList:[],workList_cur:0,memInfo:[],memSwitch:!0,inputJobId:0,inputMemSize:0,inputReqType:0,operationLog:"\n准备进行动态内存分配模拟\n",uniqIdx:0}},created:function(){var t;for(this.workList=this.dynamicMemAllocProp.memRequests,t=0;t<this.workList.length;t++)this.dynamicWorkList.push({jobId:this.workList[t].jobId,reqType:this.workList[t].reqType,reqSize:this.workList[t].reqSize,uniqId:t}),this.uniqIdx=t},computed:{switchFrame:function(){return this.frameComponentList[this.frameComponentActive]}},methods:{inputChange:function(){this.inputJobId=this.inputJobId.replace(/[^\d]/g,""),this.inputMemSize=this.inputMemSize.replace(/[^\d]/g,"")},insertRequest:function(t,e,i){this.workList.push({jobId:t,reqType:e,reqSize:i,uniqId:this.uniqIdx}),this.dynamicWorkList.push({jobId:t,reqType:e,reqSize:i,uniqId:this.uniqIdx}),this.uniqIdx++},appendLog:function(t){this.operationLog=this.operationLog+t+"\n"},algoChangeHandler:function(t){this.usingAlgo=t[0],console.log("Algorithm Switched")},functionClickHandler:function(t){this.frameComponentActive=t},executeClickHandler:function(){this.memSwitch=!this.memSwitch,0==this.usingAlgo?this.firstfit():this.bestfit()},bestfit:function(){if(this.workList_cur==this.workList.length)return 0;var t,e=Array(),i=Array(),n=this.workList[this.workList_cur].reqSize;if(this.appendLog("处理作业"+this.workList[this.workList_cur].jobId+"的"+(this.workList[this.workList_cur].reqType?"申请":"释放")+"请求-长度"+n+" B"),1==this.workList[this.workList_cur].reqType){for(s=0;s<this.memInfo.length;s++)e.push(this.memInfo[s].begin),i.push(this.memInfo[s].end);e.push(this.dynamicMemAllocProp.memContainer.maxMemory),i.push(0),e.sort(),i.sort();for(var o=!1,r=9999999999,s=0;s<=this.memInfo.length;s++)e[s]-i[s]>=n&&e[s]-i[s]<r&&(o=!0,t=i[s],r=e[s]-i[s]);if(o)for(this.dynamicWorkList.splice(0,1),this.appendLog("作业"+this.workList[this.workList_cur].jobId+"已经放入内存，起始地址:"+t+",终止地址:"+(t+n)),this.memInfo.push({begin:t,end:t+n,jobId:this.workList[this.workList_cur].jobId,orderId:0}),this.workList_cur++,this.memInfo.sort(this.firstfitSortCompare),s=0;s<this.memInfo.length;s++)this.memInfo[s].orderId=s;else this.appendLog("作业"+this.workList[this.workList_cur].jobId+"目前无法放入内存中，可用空间不足")}else{var a=!1;for(s=0;s<this.memInfo.length;s++)if(this.memInfo[s].jobId==this.workList[this.workList_cur].jobId){this.memInfo.splice(s,1),this.dynamicWorkList.splice(0,1),this.workList_cur++,a=!0;break}for(a||this.appendLog("作业"+this.workList[this.workList_cur].jobId+"未在内存中，无法进行释放。"),s=0;s<this.memInfo.length;s++)this.memInfo[s].orderId=s}},firstfitSortCompare:function(t,e){return t.begin<e.begin},firstfit:function(){if(this.workList_cur==this.workList.length)return 0;var t,e=Array(),i=Array(),n=this.workList[this.workList_cur].reqSize;if(this.appendLog("处理作业"+this.workList[this.workList_cur].jobId+"的"+(this.workList[this.workList_cur].reqType?"申请":"释放")+"请求-长度"+n+" B"),1==this.workList[this.workList_cur].reqType){for(r=0;r<this.memInfo.length;r++)e.push(this.memInfo[r].begin),i.push(this.memInfo[r].end);e.push(this.dynamicMemAllocProp.memContainer.maxMemory),i.push(0),e.sort(),i.sort();for(var o=!1,r=0;r<=this.memInfo.length;r++)if(e[r]-i[r]>=n){o=!0,t=i[r];break}if(o)for(this.dynamicWorkList.splice(0,1),this.appendLog("作业"+this.workList[this.workList_cur].jobId+"已经放入内存，起始地址:"+t+",终止地址:"+(t+n)),this.memInfo.push({begin:t,end:t+n,jobId:this.workList[this.workList_cur].jobId,orderId:0}),this.workList_cur++,this.memInfo.sort(this.firstfitSortCompare),r=0;r<this.memInfo.length;r++)this.memInfo[r].orderId=r;else this.appendLog("作业"+this.workList[this.workList_cur].jobId+"目前无法放入内存中，可用空间不足")}else{var s=!1;for(r=0;r<this.memInfo.length;r++)if(this.memInfo[r].jobId==this.workList[this.workList_cur].jobId){this.memInfo.splice(r,1),this.dynamicWorkList.splice(0,1),this.workList_cur++,s=!0;break}for(s||this.appendLog("作业"+this.workList[this.workList_cur].jobId+"未在内存中，无法进行释放。"),r=0;r<this.memInfo.length;r++)this.memInfo[r].orderId=r}}}},o=(i("ea01"),Object(r.a)(o,function(){var e=this,t=e.$createElement,t=e._self._c||t;return t("div",{staticClass:"oshw2_dynamic_mem_alloc"},[t("span",{staticClass:"oshw2_dynamic_mem_alloc_title",domProps:{innerHTML:e._s(e.dynamicMemAllocProp.title)}}),t("br"),t("div",{attrs:{id:"mem_container"}},[t("memory-container",{attrs:{memContainerProp:e.dynamicMemAllocProp.memContainer,memInfoArg:e.memInfo,memWatch:e.memSwitch}})],1),t("cust-button",{attrs:{caption:e.dynamicMemAllocProp.executeBtn.caption,selectedProp:!1},on:{cust_click:e.executeClickHandler}}),t("span",{staticStyle:{"font-size":"32px",display:"inline-block",width:"25px"}},[e._v("|")]),t("cust-button",{attrs:{caption:"查看请求列表",selectedProp:0==e.frameComponentActive},on:{cust_click:function(t){return e.functionClickHandler(0)}}}),t("cust-button",{attrs:{caption:"查看操作记录",selectedProp:1==e.frameComponentActive},on:{cust_click:function(t){return e.functionClickHandler(1)}}}),t("cust-button",{attrs:{caption:"增加作业请求",selectedProp:2==e.frameComponentActive},on:{cust_click:function(t){return e.functionClickHandler(2)}}}),t("span",{staticStyle:{"font-size":"32px",display:"inline-block",width:"25px"}},[e._v("|")]),t("cust-select",{staticStyle:{"z-index":"99999"},on:{cust_click:e.algoChangeHandler}}),t("div",{staticClass:"oshw2_dynamic_mem_alloc_frame"},[2!=e.frameComponentActive?t("div",[t(e.switchFrame,{tag:"component",attrs:{allocReqTableProp:e.dynamicWorkList,recordInfo:{log:e.operationLog}}})],1):e._e(),2==e.frameComponentActive?t("div",{staticClass:"oshw2_dynamic_mem_alloc_app"},[t("form",[t("label",[e._v("作业编号:")]),t("div",{staticClass:"oshw2_dynamic_mem_alloc_appic"},[t("input",{directives:[{name:"model",rawName:"v-model.number",value:e.inputJobId,expression:"inputJobId",modifiers:{number:!0}}],staticClass:"oshw2_dynamic_mem_alloc_appi",domProps:{value:e.inputJobId},on:{keyup:e.inputChange,input:function(t){t.target.composing||(e.inputJobId=e._n(t.target.value))},blur:function(t){return e.$forceUpdate()}}}),t("br")]),t("br"),t("label",[e._v("内存大小:")]),t("div",{staticClass:"oshw2_dynamic_mem_alloc_appic"},[t("input",{directives:[{name:"model",rawName:"v-model.number",value:e.inputMemSize,expression:"inputMemSize",modifiers:{number:!0}}],staticClass:"oshw2_dynamic_mem_alloc_appi",domProps:{value:e.inputMemSize},on:{keyup:e.inputChange,input:function(t){t.target.composing||(e.inputMemSize=e._n(t.target.value))},blur:function(t){return e.$forceUpdate()}}}),e._v(" B"),t("br")]),t("br"),t("label",[e._v("请求方式:")]),t("div",{staticClass:"oshw2_dynamic_mem_alloc_appic"},[t("input",{directives:[{name:"model",rawName:"v-model.number",value:e.inputReqType,expression:"inputReqType",modifiers:{number:!0}}],attrs:{name:"new_req_type",id:"nrt1",type:"radio",value:"1"},domProps:{checked:e._q(e.inputReqType,e._n("1"))},on:{change:function(t){e.inputReqType=e._n("1")}}}),t("label",{attrs:{for:"nrt1"}},[e._v("申请")]),t("input",{directives:[{name:"model",rawName:"v-model.number",value:e.inputReqType,expression:"inputReqType",modifiers:{number:!0}}],attrs:{name:"new_req_type",id:"nrt2",type:"radio",value:"0"},domProps:{checked:e._q(e.inputReqType,e._n("0"))},on:{change:function(t){e.inputReqType=e._n("0")}}}),t("label",{attrs:{for:"nrt2"}},[e._v("释放")])]),t("br"),t("br")]),t("cust-button",{attrs:{caption:"增加指令"},on:{cust_click:function(t){return e.insertRequest(e.inputJobId,e.inputReqType,e.inputMemSize)}}})],1):e._e()])],1)},[],!1,null,"b7623214",null)).exports,o={name:"MainContainer",props:{mainContainerProp:Object},components:{DynamicMemAlloc:o}},o={name:"App",data:function(){return{projectTitle:"Hello World",mainContainerProp:{dynamicMemAlloc:{title:"动态内存分配模拟",executeBtn:{caption:"执行"},addBtn:{caption:"增加"},memContainer:{maxMemory:655360},memRequests:[{jobId:1,reqType:1,reqSize:133120},{jobId:2,reqType:1,reqSize:61440},{jobId:3,reqType:1,reqSize:102400},{jobId:2,reqType:0,reqSize:61440},{jobId:4,reqType:1,reqSize:204800},{jobId:3,reqType:0,reqSize:102400},{jobId:1,reqType:0,reqSize:133120},{jobId:5,reqType:1,reqSize:143360},{jobId:6,reqType:1,reqSize:61440},{jobId:7,reqType:1,reqSize:51200},{jobId:6,reqType:0,reqSize:61440}]}}}},components:{TopBar:s,MainContainer:(i("e887"),Object(r.a)(o,function(){var t=this.$createElement,t=this._self._c||t;return t("div",{staticClass:"oshw2_main_container"},[t("dynamic-mem-alloc",{attrs:{dynamicMemAllocProp:this.mainContainerProp.dynamicMemAlloc}})],1)},[],!1,null,"2a44ec0e",null)).exports}},u=(i("034f"),Object(r.a)(o,function(){var t=this.$createElement,t=this._self._c||t;return t("div",{attrs:{id:"app"}},[t("top-bar",{attrs:{title:this.projectTitle}}),t("main-container",{attrs:{mainContainerProp:this.mainContainerProp}})],1)},[],!1,null,null,null)).exports;n.a.config.productionTip=!1,new n.a({render:t=>t(u)}).$mount("#app")},"692b":function(t,e,i){},"69e8":function(t,e,i){},"6f78":function(t,e,i){},"85ec":function(t,e,i){},"89ff":function(t,e,i){"use strict";i("6f78")},"92ea":function(t,e,i){},ab2c:function(t,e,i){},b23a:function(t,e,i){},b83d:function(t,e,i){"use strict";i("b23a")},e887:function(t,e,i){"use strict";i("69e8")},ea01:function(t,e,i){"use strict";i("ab2c")}});