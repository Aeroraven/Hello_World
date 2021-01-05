// ==UserScript==
// @name         Mooc Fast Mutual Judge
// @version      0.1.1
// @description  Judge as quickly as you can
// @author       Aeroraven
// @match        https://www.icourse163.org/learn/*
// @match        http://www.icourse163.org/learn/*
// @grant        none
// ==/UserScript==

(function() {
    'use strict';

    // Your code here...

    var s=document.createElement("span");
    s.style.zIndex=99999;
    s.innerHTML="Complete Judge";
    s.style.position="fixed";
    s.style.left=0;
    s.style.top=0;
    s.style.fontSize="17px";
    s.addEventListener("click",function(){
        var a=document.getElementsByClassName("j-select");
        var i=0;
        for( i=0;i<a.length;i++){
            a[i].checked=true;
        }
        for( i=0;i<a.length;i++){
            if(Math.random()<0.5){a[i].checked=true;}
        }
        var b=document.getElementsByClassName("inputtxt");
        for( i=1;i<b.length;i++){
            b[i].value="答案正确";
        }
    });
    document.getElementsByTagName("body")[0].appendChild(s);
})();
