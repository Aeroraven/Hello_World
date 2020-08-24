function fuckChaoxing(){
	this.start=function(){
		fuckChaoxing.playTimer=setInterval(function(){
			var el1=document.getElementById("iframe").contentWindow;
			var el2=el1.document.getElementsByTagName("iframe")[0].contentWindow;
			var playerObject=el2.document.getElementById("video_html5_api");
			playerObject.play();
			playerObject.playbackRate=15;
			playerObject.muted=true;
		},100);
	}
	this.terminate=function(){
		clearInterval(fuckChaoxing.playTimer);
		alert("Autoplay Stopped");
	}
	this.stop=function(){
		clearInterval(fuckChaoxing.playTimer);
		playerObject.pause();
		alert("Video Paused");
	}
	this.inject=function(){
		var newBar = document.createElement("div");
		var referNode = document.getElementsByClassName("bigImg")[0];
		var parentNode = referNode.parentNode;
		newBar.innerHTML="<div id='fkchaoxing-container' style='padding-left:15px;padding-right:15px;font-size:18px;position:fixed;top:0px;z-index:1000;background-color:#333333;border:4px solid #333333; color:#ff9900; border-radius:5px; margin-left:2px;margin-top:2px;'><span style='font-size:25px' id='fkchaoxing-title'>Chaoxing Fucker</span></div>";
		parentNode.insertBefore(newBar,referNode);
		var el=document.getElementById("fkchaoxing-title");
		parentNode = el.parentNode;
		
		var nl=document.createElement("br");
		parentNode.appendChild(nl);
		nl=document.createElement("br");
		parentNode.appendChild(nl);
		
		nl=document.createElement("a");
		nl.innerHTML="&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp";
		parentNode.appendChild(nl);
		
		nl=document.createElement("a");
		nl.onclick=function(){
			var t=new fuckChaoxing();
			t.start();
		}
		nl.href="###";
		nl.innerHTML="[ Start Autoplay ]";
		nl.style.color="#ff9900";
		parentNode.appendChild(nl);
		
		nl=document.createElement("a");
		nl.innerHTML="&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;";
		
		parentNode.appendChild(nl);
		
		nl=document.createElement("a");
		nl.onclick=function(){
			var t=new fuckChaoxing();
			t.terminate();
		}
		nl.href="###";
		nl.innerHTML="[ Stop Autoplay ]";
		nl.style.color="#ff9900";
		parentNode.appendChild(nl);
		
		
		
	}
}
function ChaoxingGetFucked(){
	var s=window.location.host;
	if(s.indexOf("chaoxing.com")!=-1){
		var fk = new fuckChaoxing();
		fk.inject();
	}
}
ChaoxingGetFucked();
