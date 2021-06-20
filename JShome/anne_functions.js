	var Download =
	{
		click : function(node) {
			var ev = document.createEvent("MouseEvents");
			ev.initMouseEvent("click", true, false, self, 0, 0, 0, 0, 0, false, false, false, false, 0, null);
			return node.dispatchEvent(ev);
		},
		encode : function(data) {
				return 'data:application/octet-stream;base64,' + btoa( data );
		},
		link : function(data, name){
			var a = document.createElement('a');
			a.download = name || self.location.pathname.slice(self.location.pathname.lastIndexOf('/')+1);
			a.href = data || self.location.href;
			return a;
		}
	};
	Download.save = function(data, name)
	{
		this.click(
			this.link(
				this.encode( data ),
				name
			)
		);
	};


	function chkForNumbers(myList) {
		// return true if one or more items in the list starts with a number
		result = false ;
		myList.forEach(function(s) {	if (s.match(/^\d/)) result = true ;}) ;
		return result ;
	}

	function IsNumeric(dataTable) {
		// return false if one or more items in the list is not a Number
		result = true ;
		rows = dataTable.split('\n') ;
		rows.shift() ;  // remove the header
		rows.forEach(function(row) {
			items = row.split('\t');
			if (items.length > 1) { // only check values if dataTable contain multiple columns
				items.shift(); // remove row name
				items.forEach(function(item) {	if (isNaN(item) || item == '' ) result = false ;}) ; // non numeric value is found
			}
		});
		return result ;
	}

	function hasDuplicates(arr) {
		return !arr.every(num => arr.indexOf(num) === arr.lastIndexOf(num));
	}

	function checkDuplicates(dataTable) {
		// check if the first column (the keys) does not contain replicates
		var keys = [] ;
		var lines = dataTable.split('\n') ;
		lines.forEach( function(line) {
			var items = line.split('\t');
			if (items.length>1) { keys.push(items[0]) ; }
		});
		return hasDuplicates(keys) ;
	}


function save_SVG2PNG(filename) {
	// save the D3 svg graph to PNG
	var svg = d3.select("svg");
	var svgString = getSVGString(svg.node());
	svgString2Image( svgString, 1400, 400, 'png', save ); // passes Blob and filesize String to the callback
	function save( dataBlob, filesize ){
		saveAs( dataBlob, filename ); // FileSaver.js function
	}
}


// Below are the functions that handle actual exporting:
// getSVGString ( svgNode ) and svgString2Image( svgString, width, height, format, callback )

function getSVGString( svgNode ) {
	svgNode.setAttribute('xlink', 'http://www.w3.org/1999/xlink');
	var cssStyleText = getCSSStyles( svgNode );
	appendCSS( cssStyleText, svgNode );
	var serializer = new XMLSerializer();
	var svgString = serializer.serializeToString(svgNode);
	svgString = svgString.replace(/(\w+)?:?xlink=/g, 'xmlns:xlink='); // Fix root xlink without namespace
	svgString = svgString.replace(/NS\d+:href/g, 'xlink:href'); // Safari NS namespace fix
	return svgString;
	function getCSSStyles( parentElement ) {
		var selectorTextArr = [];
		// Add Parent element Id and Classes to the list
		selectorTextArr.push( '#'+parentElement.id );
		for (var c = 0; c < parentElement.classList.length; c++)
				if ( !contains('.'+parentElement.classList[c], selectorTextArr) )
					selectorTextArr.push( '.'+parentElement.classList[c] );
		// Add Children element Ids and Classes to the list
		var nodes = parentElement.getElementsByTagName("*");
		for (var i = 0; i < nodes.length; i++) {
			var id = nodes[i].id;
			if ( !contains('#'+id, selectorTextArr) )
				selectorTextArr.push( '#'+id );
			var classes = nodes[i].classList;
			for (var c = 0; c < classes.length; c++)
				if ( !contains('.'+classes[c], selectorTextArr) )
					selectorTextArr.push( '.'+classes[c] );
		}
		// Extract CSS Rules
		var extractedCSSText = "";
		for (var i = 0; i < document.styleSheets.length; i++) {
			var s = document.styleSheets[i];

			try {
			    if(!s.cssRules) continue;
			} catch( e ) {
		    		if(e.name !== 'SecurityError') throw e; // for Firefox
		    		continue;
		    	}
			var cssRules = s.cssRules;
			for (var r = 0; r < cssRules.length; r++) {
				if ( contains( cssRules[r].selectorText, selectorTextArr ) )
					extractedCSSText += cssRules[r].cssText;
			}
		}

		return extractedCSSText;
		function contains(str,arr) {
			return arr.indexOf( str ) === -1 ? false : true;
		}
	}
	function appendCSS( cssText, element ) {
		var styleElement = document.createElement("style");
		styleElement.setAttribute("type","text/css");
		styleElement.innerHTML = cssText;
		var refNode = element.hasChildNodes() ? element.children[0] : null;
		element.insertBefore( styleElement, refNode );
	}
}

function svgString2Image( svgString, width, height, format, callback ) {
	var format = format ? format : 'png';
	var imgsrc = 'data:image/svg+xml;base64,'+ btoa( unescape( encodeURIComponent( svgString ) ) ); // Convert SVG string to data URL
	var canvas = document.createElement("canvas");
	var context = canvas.getContext("2d");
	canvas.width = width;
	canvas.height = height;
	var image = new Image();
	image.onload = function() {
		context.clearRect ( 0, 0, width, height );
		context.drawImage(image, 0, 0, width, height);
		canvas.toBlob( function(blob) {
			var filesize = Math.round( blob.length/1024 ) + ' KB';
			if ( callback ) callback( blob, filesize );
		});

	};
	image.src = imgsrc;
}

	function Copy2Clipboard(my_type) {
		if (my_type == 'DNA') {
			document.getElementById("ResultText").select();
		} else {
			document.getElementById("ResultText2").select();
		}
		document.execCommand("copy");
	}

	function savetextarea(){
		var textToSave = document.getElementById("ResultText").value;
		var textToSaveAsBlob = new Blob([textToSave], {type:"text/plain"});
		var textToSaveAsURL = window.URL.createObjectURL(textToSaveAsBlob);
		var downloadLink = document.createElement("a");
		downloadLink.download = "Results.txt";
		downloadLink.innerHTML = "Download File";
		downloadLink.href = textToSaveAsURL;
		downloadLink.onclick = destroyClickedElement;
		downloadLink.style.display = "none";
		document.body.appendChild(downloadLink);
		downloadLink.click();
	}
