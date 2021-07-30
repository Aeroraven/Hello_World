// ==UserScript==
// @name         Donkey Tree
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  try to take over the world!
// @author       Cesium Chloride
// @match        https://qah5.zhihuishu.com/qa.html*
// ==/UserScript==

(function() {
    'use strict';
unsafeWindow.s2 =function(){
    var x=document.getElementsByClassName("answer-li");
    for(var i=0;i<x.length;i++){
        x[i].setAttribute('id',"fuckli-"+i);
        x[i].innerHTML+="<a onclick='window.fuck(this.parentNode.children[2].children[0].children[0]);'> [直接粘贴到答题区] </a>";
    }
}
unsafeWindow.fuck=function (x){
 let text = x.innerHTML;
    navigator.clipboard.writeText(text.slice(7).substring(0,text.length-14))
        .then(() => {
        //alert('Text copied to clipboard');
    })
        .catch(err => {
        alert('Error in copying text: ', err);
    });
    document.getElementsByTagName("textarea")[0].innerHTML=text.slice(7).substring(0,text.length-14);
}
unsafeWindow.applyShit=function(){
    document.getElementsByClassName("el-dialog__header")[0].children[0].children[1].innerHTML="我来回答 <a onclick='window.fuckPaste()'>OvO</a>"
}
unsafeWindow.fuckPaste=function(){
    //document.getElementsByTagName("textarea")[0].innerHTML= (event.clipboardData || window.clipboardData).getData('text');
}
unsafeWindow.fl = 0;
unsafeWindow.t = setTimeout(function(){
    unsafeWindow.s2();
    console.log("Loaded");
},1000);

unsafeWindow.t3=setInterval(function(){
    if(document.getElementsByClassName("my-answer-btn").length!=0){
        document.getElementsByClassName("my-answer-btn")[0].click();
        clearInterval(unsafeWindow.t3);
        unsafeWindow.t4=setInterval(function(){
            if(document.getElementsByClassName("iconguanbi").length!=0){

                document.getElementsByClassName("iconguanbi")[0].click();
                clearInterval(unsafeWindow.t4);
            }
        },100);
    }
},100);
unsafeWindow.t2 =setInterval(function(){
    if(document.getElementsByClassName("el-dialog__header").length!=0){
        unsafeWindow.applyShit();
    }
    console.log("Applied");
},2500);

    // Your code here...
})();
