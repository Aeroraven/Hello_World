function fuckChaoxing(){
	
	this.terminate=function(){
		fuckChaoxing.startStatus=0;
		clearInterval(fuckChaoxing.playTimer);
		alert("Autoplay Stopped");
	}
	this.stop=function(){
		fuckChaoxing.startStatus=0;
		clearInterval(fuckChaoxing.playTimer);
		playerObject.pause();
		alert("Video Paused");
	}
	this.nextPage=function(){
		var el=document.getElementById("right2");
		el.click();
	}
	this.start=function(){
		if(fuckChaoxing.startStatus==1){
			alert("Autoplay has already been started!");
			return 0;
		}
		fuckChaoxing.playTimeCounter = 0;
		fuckChaoxing.startStatus = 1;
		fuckChaoxing.beginTime=new Date().getTime()/1000;
		fuckChaoxing.playTimer=setInterval(function(){
			var el1=document.getElementById("iframe").contentWindow;
			var el2=el1.document.getElementsByTagName("iframe")[0].contentWindow;
			var playerObject=el2.document.getElementById("video_html5_api");
			if(playerObject!=null){
				playerObject.play();
				playerObject.playbackRate=15;
				playerObject.muted=true;
				playerObject.style.filter="grayscale(100%) blur(100px) contrast(0%)";
				playerObject.parentNode.style.display="none";
				var tp=el2.document.getElementById("fkChaoxing-eyePurify");
				if(tp==null){
					var newBar = document.createElement("div");
					var parentNode = playerObject.parentNode.parentNode;
					newBar.innerHTML="<div id='fkChaoxing-eyePurify' style='background-color:#333333;color:#ff9900;padding-left:5px;padding-top:4px;'></div>";
					parentNode.insertBefore(newBar,playerObject.parentNode);
					console.log("CREATE");
					
				}else{
					var curTs=new Date().getTime()/1000;
					var remain=(curTs-fuckChaoxing.beginTime)/playerObject.currentTime*(playerObject.duration-playerObject.currentTime)
					tp.innerHTML="<span style='font-size:30px;'>"+parseInt(playerObject.currentTime/playerObject.duration*100)+"% Completed</span><br/><br/> <span style='font-size:20px'>Remaining: &nbsp;"+parseInt(remain)+" s</span> <br/>"+"<br/><br/> We automatically blocked the video player interface in order to purify your eyes.<br/><br/>";
				}
			}
			var el3=el1.document.getElementsByClassName("ans-job-icon")[0];
			if(el3!=null){
				if(getComputedStyle(el3,null)['backgroundPositionY']=="-24px"){
					var el=document.getElementById("right2");
					fuckChaoxing.beginTime=new Date().getTime()/1000;
					el.click();
					console.log("nextPage");
				}
			}
			var el4=el2.document.getElementById("ext-comp-1035");
			if(el4!=null){
				el4.innerHTML="";
			}
		},100);
	}
	this.inject=function(){
		var newBar = document.createElement("div");
		var referNode = document.getElementsByClassName("bigImg")[0];
		var parentNode = referNode.parentNode;
		newBar.innerHTML="<div id='fkchaoxing-container' style='padding-left:15px;padding-right:15px;font-size:18px;position:fixed;top:0px;z-index:1000;background-color:#333333;border:4px solid #333333; color:#ff9900; border-radius:5px; margin-left:2px;margin-top:2px;'><span style='font-size:25px' id='fkchaoxing-title'>Fucking Chaoxing </span></div>";
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
		
		nl=document.createElement("a");
		nl.innerHTML="&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp";
		parentNode.appendChild(nl);
		
		
		
		nl=document.createElement("a");
		nl.onclick=function(){
			var t=new fuckChaoxing();
			t.nextPage();
		}
		nl.href="###";
		nl.innerHTML="[ Next Chapter ]";
		nl.style.color="#ff9900";
		parentNode.appendChild(nl);
		
	}
}
function ChaoxingGetFucked(){
	var s=window.location.host;
	if(s.indexOf("chaoxing.com")!=-1){
		var fk = new fuckChaoxing();
		fuckChaoxing.startStatus=0;
		fk.inject();
	}
}
ChaoxingGetFucked();
